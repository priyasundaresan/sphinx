from dataclasses import dataclass
import time
import os
import torch
import numpy as np
import pyrallis
import cv2

import common_utils
from common_utils.eval_utils import get_reference_initial_obs, align_env_to_image, check_for_interrupt

from scripts.train_hydra import load_model

from envs.franka_env import FrankaEnv, FrankaEnvConfig

from interactive_scripts.real_pcl_extractor import RealPclExtractor
from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode

class InteractiveBot:
    def __init__(
        self,
        control_freq,
        hydra_policy,
        hydra_dataset, 
        robot_cfg,
        reference_rollout_dir,
    ):
        self.control_freq = control_freq
        self.hydra_policy = hydra_policy
        self.hydra_dataset = hydra_dataset
        self.env = FrankaEnv(robot_cfg)

        self.recorder = DatasetRecorder('hydra_real_rollouts')
        self.reference_rollout_dir = reference_rollout_dir

        self.stopwatch = common_utils.Stopwatch()

        # warm up camera for 10s
        for _ in range(int(self.control_freq) * 10):
            with common_utils.FreqGuard(self.control_freq):
               self.env.observe()

    def reset(self):
        # First open gripper
        proprio = self.env.observe_proprio()
        ee_pos = proprio.eef_pos
        ee_euler = proprio.eef_euler
        self.env.move_to(ee_pos, ee_euler, 1.0, control_freq=self.control_freq)
        # Then reset
        self.env.reset()

    def run_episode(
        self,
        vis: bool,
    ):
        mode = ActMode.Waypoint.value
        cached_dense_actions = []

        # We can align the initial state of the environment given a reference rollout
        if self.reference_rollout_dir:
            episode_idx = self.recorder.get_next_idx()
            assert(episode_idx < len([fn for fn in os.listdir(self.reference_rollout_dir) if 'npz' in fn]))
            initial_image = get_reference_initial_obs(episode_idx, self.reference_rollout_dir)
            align_env_to_image(self.env, initial_image)

        while mode != ActMode.Terminate.value:
            with common_utils.FreqGuard(self.control_freq):

                ### manual termination ###
                if check_for_interrupt():
                    print("Rollout interrupted by user.")
                    break
                ###

                obs = self.env.observe()

                if vis:
                    images = [obs[view] for view in self.hydra_dataset.camera_views]
                    stacked = np.hstack(images)
                    stacked = cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR)
                    cv2.imshow("img", stacked)
                    cv2.waitKey(1)


                ### hydra obs preprocessing ###
                hydra_obs = self.hydra_dataset.process_observation(obs)
                for k, v in hydra_obs.items():
                    hydra_obs[k] = v.cuda()
                ###

                ### hydra inference ###
                if not len(cached_dense_actions):
                    dense_action_seq, waypoint_action, mode_probs = self.hydra_policy.act(hydra_obs)
                    mode_probs = mode_probs.detach().cpu().numpy()
                    mode = mode_probs[:ActMode.Terminate.value].argmax() # FIXME, predicting fp early terminates so this is a patch

                    for dense_action in dense_action_seq.split(1, dim=0):
                        cached_dense_actions.append(dense_action.squeeze(0))
                ###

                ### execute waypoint mode ###
                if mode == ActMode.Waypoint.value:
                    ee_pos, ee_euler, gripper_open = waypoint_action.split([3, 3, 1])
                    gripper_open = 0 if (gripper_open.item() < 0.5) else 1
                    self.env.move_to(ee_pos.numpy(), ee_euler.numpy(), gripper_open, control_freq=self.control_freq, recorder=self.recorder)
                    cached_dense_actions = []
                ###

                ### execute dense mode ###
                else:
                    dense_action = cached_dense_actions.pop(0)
                    ee_pos, ee_euler, gripper_open = dense_action.split([3, 3, 1])
                    dense_action = np.concatenate([ee_pos, ee_euler, [gripper_open.item()]]).astype(np.float32)
                    self.recorder.record(ActMode.Dense, obs, dense_action)
                    self.env.apply_action(ee_pos.numpy(), ee_euler.numpy(), gripper_open.item(), is_delta=True)
                ###

        self.recorder.end_episode(save=True)
        #self.recorder.end_episode(save=False)
    
@dataclass
class EvalConfig:
    hydra_weight: str = ""
    vis: int = 1
    freq: float = 10
    reference_rollout_dir: str = "" # Optionally pass a reference directory to load initial env states from

def main(cfg: EvalConfig):

    hydra_policy, hydra_dataset, train_cfg = load_model("%s/latest.pt"%cfg.hydra_weight, "cuda", load_only_one=True)
    hydra_policy.eval()
    env_cfg_path = os.path.join(train_cfg.dataset.path, "env_cfg.yaml")
    env_cfg = pyrallis.load(FrankaEnvConfig, open(env_cfg_path))  # type: ignore

    if hydra_policy.cfg.use_ddpm:
        common_utils.cprint(f"Warning: override to use ddim with step 10")
        hydra_policy.cfg.use_ddpm = 0
        hydra_policy.cfg.ddim.num_inference_timesteps = 10

    agent = InteractiveBot(cfg.freq, hydra_policy, hydra_dataset, env_cfg, cfg.reference_rollout_dir)
    for i in range(10):
        agent.reset()
        agent.run_episode(cfg.vis)

if __name__ == "__main__":
    """example command:

    python scripts/eval_hydra.py --hydra_weight exps/hydra/cups2 --env_cfg_path envs/fr3.yaml --freq 10 --reference_rollout_dir ours_real_rollouts
    """
    import rich.traceback

    rich.traceback.install()
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)

    cfg = pyrallis.parse(config_class=EvalConfig)  # type: ignore
    main(cfg)
