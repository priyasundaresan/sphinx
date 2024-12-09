from dataclasses import dataclass, field
import os
import sys
import yaml
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import diffusers
import pyrallis
import numpy as np

import common_utils
from dataset_utils.waypoint_dataset import PointCloudDataset, PointCloudDatasetConfig
from models.waypoint_transformer import WaypointTransformer, WaypointTransformerConfig
from models.pointnet2_utils import farthest_point_sample


def generate_points_mask(
    clicked_labels: torch.Tensor, pred_clicked_logits: torch.Tensor, topk_from_pred: int
):
    """
    args:
        pred_clicked_logits: [batch, num_point]
        clicked_labels: [batch, num_point]
    """
    mask = clicked_labels.clone()
    if topk_from_pred == 0:
        return mask

    _, ranked_pred_clicked_indices = pred_clicked_logits.sort(dim=1, descending=True)
    # [batch, topk_from_pred]
    pred_indices_to_take = ranked_pred_clicked_indices[:, :topk_from_pred]
    mask.scatter_(1, pred_indices_to_take, 1)
    return mask


def run_one_epoch(
    policy: WaypointTransformer,
    optim: torch.optim.Optimizer,
    grad_clip: float,
    dataloader,
    stat: common_utils.MultiCounter,
    stopwatch: common_utils.Stopwatch,
    ema_policy: Optional[common_utils.EMA],
    total_optim_step: int,
):
    assert policy.training

    dataiter = iter(dataloader)
    while True:
        try:
            with stopwatch.time("data"):
                (
                    points,
                    proprio,
                    user_clicked_labels,
                    action_pos,
                    action_rot,
                    action_gripper,
                    target_mode,
                ) = next(dataiter)
        except StopIteration:
            break

        with stopwatch.time("fps"):
            points: torch.Tensor = points.cuda()
            proprio: torch.Tensor = proprio.cuda()
            user_clicked_labels: torch.Tensor = user_clicked_labels.float().cuda()

            # first fps
            if points.size(1) != policy.cfg.npoints:
                fps_indices = farthest_point_sample(points[:, :, :3], policy.cfg.npoints)
                points = points.gather(1, fps_indices.unsqueeze(2).repeat(1, 1, 6))
                user_clicked_labels = user_clicked_labels.gather(1, fps_indices)

        with stopwatch.time("model"):
            (
                pred_clicked_logits,
                pred_points_off,
                pred_pos,
                pred_rot,
                pred_gripper_logit,
                pred_mode_logit,
            ) = policy.forward(points, proprio)

            ### click ###
            target_user_clicked = user_clicked_labels / user_clicked_labels.sum(dim=1, keepdim=True)
            click_loss = F.cross_entropy(pred_clicked_logits, target_user_clicked, reduction="mean")

            if not policy.cfg.pred_point:
                click_loss = 0

            ### gripper ###
            action_gripper: torch.Tensor = action_gripper.cuda().round()
            gripper_loss = F.binary_cross_entropy_with_logits(pred_gripper_logit, action_gripper)

            ### mode ###
            target_mode = target_mode.cuda()
            mode_loss = F.cross_entropy(pred_mode_logit, target_mode)

            ### rot & pos ###
            # action_pos: [batch, 3], action_rot: [batch, 3 or 4]
            if policy.cfg.pred_off:
                points_xyz = points[:, :, :3]
                points_off = points_xyz - action_pos.cuda().unsqueeze(1)
                points_mask = generate_points_mask(
                    user_clicked_labels, pred_clicked_logits, policy.cfg.topk_train
                )
                stat[f"rot_pos/num_postive"].append(points_mask.sum(1).mean().item())

                assert pred_points_off.size() == points_off.size()
                points_off_loss = F.mse_loss(pred_points_off, points_off, reduction="none")
                # sum over the last dim, but average over num of valid mask
                points_off_loss = (points_off_loss.sum(2) * points_mask).sum(1) / points_mask.sum(1)
                pos_loss = points_off_loss.mean(0)
            else:
                pos_loss = F.mse_loss(pred_pos, action_pos.cuda())

            if policy.cfg.per_point_rot:
                action_rot: torch.Tensor = (
                    action_rot.unsqueeze(1).cuda().repeat(1, pred_rot.size(1), 1)
                )
                assert action_rot.size() == pred_rot.size()
                points_rot_loss = F.mse_loss(pred_rot, action_rot, reduction="none")
                points_rot_loss = (points_rot_loss.sum(2) * points_mask).sum(1) / points_mask.sum(1)
                rot_loss = points_rot_loss.mean(0)
            else:
                assert action_rot.size() == pred_rot.size()
                rot_loss = F.mse_loss(pred_rot, action_rot.cuda())

            # combine
            loss = click_loss + gripper_loss + rot_loss + pos_loss + mode_loss
            optim.zero_grad()
            loss.backward()
            if grad_clip > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)  # type: ignore
                stat[f"train/grad_norm"].append(gnorm.item())
            optim.step()
            total_optim_step += 1

            if ema_policy is not None:
                ema_decay = ema_policy.step(policy, optim_step=total_optim_step)
                stat[f"train/ema_decay"].append(ema_decay)

            torch.cuda.synchronize()

        with stopwatch.time("stat"):
            ### click ###
            if policy.cfg.pred_point:
                assert isinstance(click_loss, torch.Tensor)
                stat[f"click/loss"].append(click_loss.item())
                oracle_click_loss = -target_user_clicked * (target_user_clicked + 1e-5).log()
                oracle_click_loss = oracle_click_loss.sum(1).mean(0).item()
                stat[f"click/loss(*)"].append(click_loss.item() - oracle_click_loss)

                _, click_order = pred_clicked_logits.sort(dim=1, descending=True)
                # click_order: [batch, num_point], label: [batch, num_point]
                ordered_label = (user_clicked_labels.gather(1, click_order) > 0).float()
                for k in [3, 10, 20]:
                    topk = ordered_label[:, :k]
                    stat[f"click/acc_top{k}"].append(topk.mean().item())

            ### gripper ###
            stat[f"gripper/loss"].append(gripper_loss.item())
            stat[f"gripper/acc"].append(
                ((pred_gripper_logit > 0).float() == action_gripper).float().mean().item()
            )

            ### mode ###
            stat[f"mode/loss"].append(mode_loss.item())
            pred_mode_class = pred_mode_logit.argmax(dim=1)  # No need for softmax
            mode_accuracy = (pred_mode_class == target_mode).float().mean().item()
            stat[f"mode/acc"].append(mode_accuracy)

            ### rot & pos ###
            stat[f"rot_pos/pos_loss"].append(pos_loss.item())
            stat[f"rot_pos/rot_loss"].append(rot_loss.item())

    stat[f"train/total_optim_step"].append(total_optim_step)
    return total_optim_step


def eval_inference_err(
    policy: WaypointTransformer,
    num_pass: int,
    dataset: PointCloudDataset,
    stat: common_utils.MultiCounter,
):
    policy.train(False)

    for data in dataset.datas:
        xyz = torch.from_numpy(data["xyz"]).float()
        color = torch.from_numpy(data["xyz_color"]).float()
        proprio = torch.from_numpy(data["proprio"]).float()

        with torch.no_grad():
            pred_pos = policy.inference(xyz, color, proprio, num_pass=num_pass)[1]

        err_pos = np.sqrt(np.sum((data["action_pos"] - pred_pos) ** 2))
        stat["eval/err_pos(cm)"].append(100 * err_pos)
    return stat["eval/err_pos(cm)"].mean()


@dataclass
class MainConfig(common_utils.RunConfig):
    seed: int = 1
    # optim
    epoch: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    grad_clip: float = 1.0
    cosine_schedule: int = 0
    use_ema: int = 0
    # data & model
    dataset: PointCloudDatasetConfig = field(default_factory=PointCloudDatasetConfig)
    waypoint: WaypointTransformerConfig = field(default_factory=WaypointTransformerConfig)
    train_split: str = "train"
    num_workers: int = 2
    # log
    eval_per_epoch: int = 1
    save_per: int = -1
    num_eval_episode: int = 20
    save_dir: str = "exps/waypoint/tsfm1"
    use_wb: int = 0


def main():
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore
    common_utils.set_all_seeds(cfg.seed)

    # logging
    log_path = os.path.join(cfg.save_dir, "train.log")
    sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)
    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
    print(common_utils.wrap_ruler("config"))
    with open(cfg.cfg_path, "r") as f:
        print(f.read(), end="")

    stat = common_utils.MultiCounter(
        cfg.save_dir,
        bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=yaml.safe_load(open(cfg.cfg_path, "r")),
    )

    # policy & optim
    policy = WaypointTransformer(cfg.waypoint).cuda()
    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    if cfg.cosine_schedule:
        lr_scheduler = diffusers.get_cosine_schedule_with_warmup(optim, 0, cfg.epoch)
    else:
        lr_scheduler = diffusers.get_constant_schedule(optim)

    ema_policy = None
    if cfg.use_ema:
        ema_policy = common_utils.EMA(policy, power=3 / 4)

    # data
    train_dataset = PointCloudDataset(
        cfg.dataset, bool(policy.cfg.use_euler), policy.cfg.npoints, cfg.train_split
    )
    if cfg.dataset.fps and cfg.num_workers < 10:
        print(f"Warning: {cfg.num_workers = } is too small")

    train_dataloader = DataLoader(
        train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True
    )
    eval_dataset_cfg = copy.deepcopy(cfg.dataset)
    eval_dataset_cfg.repeat = 1
    eval_dataset_cfg.aug_interpolate = 0
    eval_dataset_cfg.aug_translate = 0
    eval_dataset = PointCloudDataset(
        eval_dataset_cfg, bool(policy.cfg.use_euler), policy.cfg.npoints, cfg.train_split
    )

    # train <[o-o-o-o-o]Z!
    stopwatch = common_utils.Stopwatch()
    common_utils.count_parameters(policy)
    saver = common_utils.TopkSaver(cfg.save_dir, 1)
    total_optim_step = 0
    for epoch in range(cfg.epoch):
        total_optim_step = run_one_epoch(
            policy,
            optim,
            cfg.grad_clip,
            train_dataloader,
            stat,
            stopwatch,
            ema_policy,
            total_optim_step,
        )
        stat["train/lr(x1000)"].append(lr_scheduler.get_last_lr()[0] * 1000)
        lr_scheduler.step()

        if (epoch + 1) % cfg.eval_per_epoch == 0:
            with stopwatch.time("eval"), common_utils.eval_mode(policy):
                eval_and_save(epoch, policy, ema_policy, eval_dataset, saver, stat, cfg)

        stat.summary(epoch + 1)
        stopwatch.summary()
        print("########################################################")


def eval_and_save(
    epoch: int,
    policy: WaypointTransformer,
    ema_policy: Optional[common_utils.EMA],
    eval_dataset: PointCloudDataset,
    saver: common_utils.TopkSaver,
    stat: common_utils.MultiCounter,
    cfg: MainConfig,
):
    pos_err = eval_inference_err(policy, 1, eval_dataset, stat)

    if eval_dataset.cfg.is_real:
        if cfg.save_per > 0 and (epoch + 1) % cfg.save_per == 0:
            force_save_name = f"epoch{epoch+1}"
        else:
            force_save_name = None

        saver.save(policy.state_dict(), -pos_err, save_latest=True, force_save_name=force_save_name)
        if ema_policy is not None:
            ema_weights = ema_policy.stable_model.state_dict()
            if force_save_name is not None:
                force_save_name += "_ema"
            saver.save(ema_weights, -pos_err, save_latest=True, force_save_name=force_save_name)
    else:
        from scripts.eval_sim_waypoint import eval_waypoint_policy

        score = eval_waypoint_policy(
            policy, eval_dataset.env_cfg_path, 3, cfg.num_eval_episode, stat
        )
        saver.save(policy.state_dict(), score, save_latest=True)

        if ema_policy is not None:
            ema_eval = ema_policy.stable_model
            ema_score = eval_waypoint_policy(
                ema_eval, eval_dataset.env_cfg_path, 3, cfg.num_eval_episode, stat, prefix="ema_"
            )
            stat["eval/ema-delta"].append(ema_score - score)
            saver.save(ema_eval.state_dict(), ema_score, force_save_name="ema")


def load_waypoint(model_path):
    train_cfg_path = os.path.join(os.path.dirname(model_path), "cfg.yaml")
    train_cfg = pyrallis.load(MainConfig, open(train_cfg_path, "r"))  # type: ignore
    env_cfg = os.path.join(train_cfg.dataset.path, "env_cfg.yaml")
    if train_cfg.dataset.is_real:
        from envs.franka_env import FrankaEnvConfig

        env_cfg = pyrallis.load(FrankaEnvConfig, open(env_cfg, "r"))  # type: ignore
    else:
        from envs.robomimic_env import RobomimicEnvConfig

        env_cfg = pyrallis.load(RobomimicEnvConfig, open(env_cfg, "r"))  # type: ignore

    policy = WaypointTransformer(train_cfg.waypoint)
    policy.load_state_dict(torch.load(model_path))

    return policy, env_cfg


if __name__ == "__main__":
    main()
