from dataclasses import dataclass, field
from collections import defaultdict
import os
import sys
import numpy as np
import torch
import yaml
import pyrallis
import diffusers

import common_utils
from interactive_scripts.dataset_recorder import ActMode
from dataset_utils.waypoint_dataset import augment_with_translation
from dataset_utils.dense_dataset import Batch
from envs.robomimic_env import SimPclExtractor
from models.pointcloud_dp import DP3, DP3Config


@dataclass
class PointCloudDatasetConfig:
    path: str = ""
    num_data: int = -1
    is_sim: int = 1
    max_num_point: int = 20000

    def __post_init__(self):
        datasets = {
            "square": "data/square3",
            "can": "data/can_crop1",
            "drawer_new2": "data/drawer_new2",
        }
        if self.path in datasets:
            self.path = datasets[self.path]


class PointCloudDataset:
    def __init__(self, cfg: PointCloudDatasetConfig, load_only_one=False):
        self.cfg = cfg
        # load_only_one makes loading faster for non-training purpose
        self.load_only_one = load_only_one

        self.episodes: list[list[dict]] = self._load_and_process_episodes(cfg.path, cfg.num_data)
        self.idx2entry = {}  # map from a single number to
        for episode_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode)):
                self.idx2entry[len(self.idx2entry)] = (episode_idx, step_idx)

        print(f"Dataset loaded from {cfg.path}")
        print(f"  episodes: {len(self.episodes)}")
        print(f"  steps: {len(self.idx2entry)}")
        print(f"  avg episode len: {len(self.idx2entry) / len(self.episodes):.1f}")

    @property
    def action_dim(self) -> int:
        return self.episodes[0][0]["action"].size(0)

    @property
    def prop_dim(self) -> int:
        return self.episodes[0][0]["prop"].size(0)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]

    def _load_and_process_episodes(self, path, num_data):
        print(f"loading data from {path}")
        npz_files = list(sorted(common_utils.get_all_files(path, "npz")))
        if self.load_only_one:
            npz_files = npz_files[:1]

        if self.cfg.is_sim:
            self.extractor = SimPclExtractor(path)
        else:
            from envs.franka_env_config import FrankaEnvConfig
            from interactive_scripts.real_pcl_extractor import RealPclExtractor
            import pyrallis

            env_cfg = os.path.join(self.cfg.path, "env_cfg.yaml")
            robot_cfg = pyrallis.load(FrankaEnvConfig, open(env_cfg, "r"))  # type: ignore
            self.extractor = RealPclExtractor(
                ["agent1", "agent2"], robot_cfg.calib, robot_cfg.min_bound, robot_cfg.max_bound
            )

        max_num_points = 0
        all_episodes: list[list[dict]] = []
        for episode_idx, f in enumerate(sorted(npz_files)):
            if num_data > 0 and episode_idx >= num_data:
                break
            success_msg = ""
            raw_episode = np.load(f, allow_pickle=True)["arr_0"]
            episode = []
            for timestep in raw_episode:
                if timestep["mode"] == ActMode.Waypoint:
                    continue

                action = timestep["action"]
                obs = timestep["obs"]
                points, colors = self.extractor.extract_pointcloud(obs)
                proprio = obs["proprio"]

                max_num_points = max(max_num_points, points.shape[0])
                # print(points.shape[0])

                processed_timestep = {
                    "action": torch.from_numpy(action).float(),
                    "points": torch.from_numpy(points),
                    "colors": torch.from_numpy(colors),
                    "prop": torch.from_numpy(proprio),
                }
                episode.append(processed_timestep)

                if not success_msg and timestep.get("reward", 0) > 0:
                    success_msg = f", success since {len(episode)}"

            print(f"episode {episode_idx}, len: {len(episode)}" + success_msg)
            all_episodes.append(episode)

        self.max_num_points = min(self.cfg.max_num_point, max_num_points)
        return all_episodes

    def get_action_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        action_max = self.episodes[0][0]["action"]
        action_min = self.episodes[0][0]["action"]

        for episode in self.episodes:
            for timestep in episode:
                action_max = torch.maximum(action_max, timestep["action"])
                action_min = torch.minimum(action_min, timestep["action"])

        print(f"raw action value range, the model should do all the normalization:")
        for i in range(len(action_min)):
            print(f"  dim {i}, min: {action_min[i].item():.5f}, max: {action_max[i].item():.5f}")

        return action_min, action_max

    def _convert_to_batch(self, samples, device):
        batch = {}
        for k, v in samples.items():
            batch[k] = torch.stack(v).to(device)

        action = {"action": batch.pop("action")}
        ret = Batch(obs=batch, action=action)
        return ret

    def _stack_actions(self, idx, begin, action_len):
        """stack actions in [begin, end)"""
        episode_idx, step_idx = self.idx2entry[idx]
        episode = self.episodes[episode_idx]

        actions = []
        valid_actions = []
        for action_idx in range(begin, begin + action_len):
            if action_idx < 0:
                actions.append(torch.zeros_like(episode[step_idx]["action"]))
                valid_actions.append(0)
            elif action_idx < len(episode):
                actions.append(episode[action_idx]["action"])
                valid_actions.append(1)
            else:
                actions.append(torch.zeros_like(actions[-1]))
                valid_actions.append(0)

        valid_actions = torch.tensor(valid_actions, dtype=torch.float32)
        actions = torch.stack(actions, dim=0)
        return actions, valid_actions

    def sample_dp(self, batchsize, action_pred_horizon, device):
        indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.idx2entry[idx]
            entry: dict = self.episodes[episode_idx][step_idx]

            points: torch.Tensor = entry["points"]
            colors: torch.Tensor = entry["colors"]
            proprio: torch.Tensor = entry["prop"]

            # pad the points
            if self.max_num_points > points.shape[0]:
                num_padding_point = self.max_num_points - points.shape[0]
                padding_indices = np.random.choice(points.shape[0], num_padding_point, replace=True)
                indices = torch.from_numpy(
                    np.concatenate((np.arange(points.shape[0]), padding_indices))
                )
            else:
                indices = np.random.choice(points.shape[0], self.max_num_points, replace=False)
                indices = torch.from_numpy(indices)

            points = points[indices]
            colors = colors[indices]

            # augmentation
            points, colors, _, proprio = augment_with_translation(
                points, colors, torch.rand(3), proprio
            )

            samples["points"].append(torch.cat([points, colors], dim=1).float())
            samples["prop"].append(proprio.float())

            # action chunking
            actions, valid_actions = self._stack_actions(idx, step_idx, action_pred_horizon)
            assert torch.equal(actions[0], entry["action"])
            samples["action"].append(actions)
            samples["valid_action"].append(valid_actions)

        return self._convert_to_batch(samples, device)


@dataclass
class MainConfig(common_utils.RunConfig):
    dataset: PointCloudDatasetConfig = field(default_factory=PointCloudDatasetConfig)
    dp: DP3Config = field(default_factory=DP3Config)
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
    num_eval: int = 100
    eval_seed: int = 99999
    # log
    save_dir: str = "exps/dense_pcd/run1"
    use_wb: int = 0


def run(cfg: MainConfig):
    print(common_utils.wrap_ruler("Train dataset"))
    dataset = PointCloudDataset(cfg.dataset)

    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
    print(common_utils.wrap_ruler("config"))
    with open(cfg.cfg_path, "r") as f:
        print(f.read(), end="")
    cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

    policy = DP3(prop_dim=dataset.prop_dim, action_dim=dataset.action_dim, cfg=cfg.dp).to("cuda")
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
                loss: torch.Tensor = policy.loss(batch, stopwatch)

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
        if cfg.dataset.is_sim and cfg.num_eval:
            score, eval_len = _eval_sim(
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


def _eval_sim(policy: DP3, dataset_path: str, eval_seed: int, num_eval: int):
    from envs.robomimic_env import RobomimicEnvConfig
    from scripts.eval_sim import run_eval_seeds

    env_cfg_path = os.path.join(dataset_path, "env_cfg.yaml")
    env_cfg = pyrallis.load(RobomimicEnvConfig, open(env_cfg_path))  # type: ignore

    seeds = list(range(eval_seed, eval_seed + num_eval))
    scores, eval_lens = run_eval_seeds(policy, env_cfg, seeds, 20, None, False)
    scores = list(scores.values())
    eval_lens = list(eval_lens.values())
    return np.mean(scores), np.mean(eval_lens)


def test_dataset():
    cfg = PointCloudDatasetConfig(path="data/square3", num_data=5)
    dataset = PointCloudDataset(cfg, load_only_one=False)

    batch = dataset.sample_dp(100, 16, "cuda")
    for k, v in batch.obs.items():
        print(k, v.size())
    print(batch.action["action"].size())


def main():
    import rich.traceback

    rich.traceback.install()
    torch.set_printoptions(linewidth=100)

    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    common_utils.set_all_seeds(cfg.seed)
    log_path = os.path.join(cfg.save_dir, "train.log")
    sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)
    run(cfg)


if __name__ == "__main__":
    # test_dataset()
    main()
