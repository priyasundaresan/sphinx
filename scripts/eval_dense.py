import os
from dataclasses import dataclass
import torch
import numpy as np
import pyrallis
import cv2

import common_utils
from common_utils.eval_utils import (
    get_reference_initial_obs,
    align_env_to_image,
    check_for_interrupt,
)


from dataset_utils.dense_dataset import DenseDataset
from scripts.train_dense import load_model
from envs.franka_env import FrankaEnv, FrankaEnvConfig
from models.diffusion_policy import DiffusionPolicy
from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode


def run_episode(
    policy: DiffusionPolicy,
    dataset: DenseDataset,
    env: FrankaEnv,
    recorder: DatasetRecorder,
    freq: float,
    show_camera: int,
    reference_rollout_dir: str,
):
    assert not policy.training
    stopwatch = common_utils.Stopwatch()

    with stopwatch.time("reset"):
        env.reset()

    # We can align the initial state of the environment given a reference rollout
    if reference_rollout_dir:
        episode_idx = recorder.get_next_idx()
        assert episode_idx < len([fn for fn in os.listdir(reference_rollout_dir) if "npz" in fn])
        initial_image = get_reference_initial_obs(episode_idx, reference_rollout_dir)
        align_env_to_image(env, initial_image)

    cached_actions = []
    mode_history = []
    mode = ActMode.Dense.value
    consecutive_modes_required = 5
    TERMINATE_THRESH = 1.3

    while mode == ActMode.Dense.value:
        with common_utils.FreqGuard(freq):
            with stopwatch.time("observe"):
                obs = env.observe()

                # Visualize current observations
                if show_camera:
                    images = [obs[cam] for cam in dataset.camera_views]
                    stacked = np.hstack(images)
                    stacked = cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR)
                    cv2.imshow("img", stacked)
                    cv2.waitKey(1)

                # Check if user interrupted
                if check_for_interrupt():
                    print("Rollout interrupted by user.")
                    break

                # Pre-process
                dense_obs = dataset.process_observation(obs)
                for k, v in dense_obs.items():
                    dense_obs[k] = v.cuda()

            # No cached actions
            if len(cached_actions) == 0:
                with stopwatch.time("act"):
                    action_seq = policy.act(dense_obs)

                for action in action_seq.split(1, dim=0):
                    cached_actions.append(action.squeeze(0))

            # Grab action
            action = cached_actions.pop(0)
            ee_pos, ee_euler, gripper_open, raw_mode = action.split([3, 3, 1, 1])

            # Only terminate if the mode flag is terminal for consecutive timesteps
            if len(mode_history) == consecutive_modes_required:
                if np.all(np.array(mode_history) > TERMINATE_THRESH):
                    mode = ActMode.Terminate.value
                else:
                    mode = ActMode.Dense.value
                mode_history = []
            mode_history.append(raw_mode.item())

            # Step action
            action = np.concatenate([ee_pos, ee_euler, [gripper_open.item()]]).astype(np.float32)
            recorder.record(ActMode.Dense, obs, action)
            env.apply_action(ee_pos.numpy(), ee_euler.numpy(), gripper_open.item(), is_delta=True)

    # End episode
    stopwatch.summary()
    recorder.end_episode(save=True)
    return


@dataclass
class EvalConfig:
    weight: str = ""
    show_camera: int = 0
    freq: float = 10
    num_episodes: int = 10
    # Optionally pass a reference directory to load initial env states from
    reference_rollout_dir: str = ""


def main(cfg: EvalConfig):
    # we also load dataset (fast loading mode with only 1 episode)
    # to get some basic info such as observation shape, action_dim
    # we also need the image preprocessing code from the dataset
    policy, dataset, train_cfg = load_model("%s/latest.pt" % cfg.weight, "cuda", load_only_one=True)
    policy.eval()
    env_cfg_path = os.path.join(train_cfg.dataset.path, "env_cfg.yaml")
    env_cfg = pyrallis.load(FrankaEnvConfig, open(env_cfg_path))  # type: ignore

    if policy.cfg.use_ddpm:
        common_utils.cprint(f"Warning: override to use ddim with step 10")
        policy.cfg.use_ddpm = 0
        policy.cfg.ddim.num_inference_timesteps = 10

    # assert False
    env = FrankaEnv(env_cfg)
    env.reset()

    # warm up camera for 10s
    for _ in range(int(cfg.freq) * 5):
        with common_utils.FreqGuard(cfg.freq):
            env.observe()

    recorder = DatasetRecorder("dense_real_rollouts")

    for _ in range(cfg.num_episodes):
        run_episode(
            policy, dataset, env, recorder, cfg.freq, cfg.show_camera, cfg.reference_rollout_dir
        )


if __name__ == "__main__":
    """example command:

    python scripts/eval_dense.py --weight exps/dense/cups2 --freq 10 --reference_rollout_dir ours_real_rollouts
    """
    import rich.traceback

    rich.traceback.install()
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)
    # torch.backends.cudnn.benchmark = False  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore

    cfg = pyrallis.parse(config_class=EvalConfig)  # type: ignore
    main(cfg)
