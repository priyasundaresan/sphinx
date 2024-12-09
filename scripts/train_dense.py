from dataclasses import dataclass, field
import os
import sys
import yaml
import pyrallis
import torch
import diffusers
import numpy as np

import common_utils
from dataset_utils.dense_dataset import DenseDataset, DenseDatasetConfig
from models.diffusion_policy import DiffusionPolicy, DiffusionPolicyConfig


@dataclass
class MainConfig(common_utils.RunConfig):
    dataset: DenseDatasetConfig = field(default_factory=DenseDatasetConfig)
    dp: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)
    norm_action: int = 1
    # training
    seed: int = 1
    num_epoch: int = 20
    epoch_len: int = 10000
    batch_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 5
    weight_decay: float = 0
    use_ema: int = 1
    ema_tau: float = 0.01
    cosine_schedule: int = 0
    lr_warm_up_steps: int = 0
    # sim eval
    is_sim: int = 0  # TODO: unify is_sim and is_real
    num_eval: int = 100
    eval_seed: int = 99999
    # log
    save_dir: str = "exps/dense/run1"
    use_wb: int = 0


def run(cfg: MainConfig):
    print(common_utils.wrap_ruler("Train dataset"))
    dataset = DenseDataset(cfg.dataset)

    if cfg.is_sim:
        from envs.robomimic_env import RobomimicEnvConfig
        from scripts.eval_sim import run_eval_seeds

    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
    print(common_utils.wrap_ruler("config"))
    with open(cfg.cfg_path, "r") as f:
        print(f.read(), end="")
    cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

    policy = DiffusionPolicy(
        obs_horizon=1,
        obs_shape=dataset.obs_shape,
        prop_dim=dataset.prop_dim,
        action_dim=dataset.action_dim,
        camera_views=dataset.camera_views,
        cfg=cfg.dp,
    ).to("cuda")
    if cfg.norm_action:
        policy.init_action_normalizer(*dataset.get_action_range())

    print(common_utils.wrap_ruler("policy weights"))
    print(policy)

    ema_policy = None
    if cfg.use_ema:
        ema_policy = common_utils.EMA(policy, power=3 / 4)

    common_utils.count_parameters(policy)
    if cfg.weight_decay == 0:
        print("Using Adam optimzer")
        optim = torch.optim.Adam(policy.parameters(), cfg.lr)
    else:
        print("Using AdamW optimzer")
        optim = torch.optim.AdamW(policy.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.cosine_schedule:
        lr_scheduler = diffusers.get_cosine_schedule_with_warmup(
            optim, cfg.lr_warm_up_steps, cfg.num_epoch * cfg.epoch_len
        )
    else:
        lr_scheduler = diffusers.get_constant_schedule(optim)

    stat = common_utils.MultiCounter(
        cfg.save_dir,
        bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=cfg_dict,
    )

    saver = common_utils.TopkSaver(cfg.save_dir, 1)
    stopwatch = common_utils.Stopwatch()
    optim_step = 0

    for _ in range(cfg.num_epoch):
        stopwatch.reset()

        for _ in range(cfg.epoch_len):
            with stopwatch.time("sample"):
                batch = dataset.sample_dp(cfg.batch_size, cfg.dp.prediction_horizon, "cuda:0")

            with stopwatch.time("train"):
                loss: torch.Tensor = policy.loss(batch)

                optim.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(  # type: ignore
                    policy.parameters(), max_norm=cfg.grad_clip
                )
                optim.step()
                lr_scheduler.step()
                torch.cuda.synchronize()

                stat["train/lr(x1000)"].append(lr_scheduler.get_last_lr()[0] * 1000)
                stat["train/loss"].append(loss.item())
                stat["train/grad_norm"].append(grad_norm.item())

                optim_step += 1
                if ema_policy is not None:
                    decay = ema_policy.step(policy, optim_step=optim_step)
                    stat["train/decay"].append(decay)

        epoch_time = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(cfg.epoch_len / epoch_time)
        policy_to_save = ema_policy.stable_model if ema_policy else policy
        if cfg.is_sim and cfg.num_eval > 0:
            score, eval_len = eval_sim(
                policy_to_save, cfg.dataset.path, cfg.eval_seed, cfg.num_eval
            )
            stat["eval/score"].append(score)
            stat["eval/eval_lens"].append(eval_len)
            metric = score
        else:
            metric = -stat["train/loss"].mean()
        stat.summary(optim_step)
        stopwatch.summary()
        saver.save(policy_to_save.state_dict(), metric, save_latest=True)

    # quit this way to avoid wandb hangs
    assert False


def eval_sim(policy: DiffusionPolicy, dataset_path: str, eval_seed: int, num_eval: int):
    from envs.robomimic_env import RobomimicEnvConfig
    from scripts.eval_sim import run_eval_seeds

    env_cfg_path = os.path.join(dataset_path, "env_cfg.yaml")
    env_cfg = pyrallis.load(RobomimicEnvConfig, open(env_cfg_path))  # type: ignore

    seeds = list(range(eval_seed, eval_seed + num_eval))
    scores, eval_lens = run_eval_seeds(policy, env_cfg, seeds, 20, None, False)
    scores = list(scores.values())
    eval_lens = list(eval_lens.values())
    return np.mean(scores), np.mean(eval_lens)


def load_model(weight_file, device, *, verbose=True, load_only_one=False):
    run_folder = os.path.dirname(weight_file)
    cfg_path = os.path.join(run_folder, f"cfg.yaml")
    if verbose:
        print(common_utils.wrap_ruler("config of loaded agent"))
        with open(cfg_path, "r") as f:
            print(f.read(), end="")
        print(common_utils.wrap_ruler(""))

    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    dataset = DenseDataset(cfg.dataset, load_only_one=load_only_one)
    policy = DiffusionPolicy(
        obs_horizon=1,
        obs_shape=dataset.obs_shape,
        prop_dim=dataset.prop_dim,
        action_dim=dataset.action_dim,
        camera_views=dataset.camera_views,
        cfg=cfg.dp,
    ).to(device)
    policy.load_state_dict(torch.load(weight_file))
    return policy, dataset, cfg


if __name__ == "__main__":
    import rich.traceback

    rich.traceback.install()
    torch.set_printoptions(linewidth=100)

    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    common_utils.set_all_seeds(cfg.seed)
    log_path = os.path.join(cfg.save_dir, "train.log")
    sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)
    run(cfg)
