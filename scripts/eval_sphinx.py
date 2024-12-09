from dataclasses import dataclass
import time
import os
import torch
import numpy as np
import pyrallis
import cv2

import common_utils

from scripts.train_dense import load_model
from scripts.train_waypoint import load_waypoint

from envs.franka_env import FrankaEnv, FrankaEnvConfig

from interactive_scripts.real_pcl_extractor import RealPclExtractor
from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode

from scripts.eval_waypoint import vis_pred

class InteractiveBot:
    def __init__(
        self,
        control_freq,
        waypoint_policy,
        dense_policy, 
        dense_dataset, 
        robot_cfg,
    ):
        self.control_freq = control_freq
        self.waypoint_policy = waypoint_policy
        self.env = FrankaEnv(robot_cfg)
        self.pcl_extractor = RealPclExtractor(["agent1", "agent2"], robot_cfg.calib, robot_cfg.min_bound, robot_cfg.max_bound)

        self.dense_policy = dense_policy
        self.dense_dataset = dense_dataset
        self.recorder = DatasetRecorder('real_rollouts')

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
        while mode != ActMode.Terminate.value:
            if mode == ActMode.Waypoint.value: 
                mode = self.run_waypoint_mode(vis=False)
            elif mode == ActMode.Dense.value:
                mode = self.run_dense_mode(vis=vis)
        self.recorder.end_episode(save=True)
    
    def run_waypoint_mode(self, vis):
        obs = self.env.observe()
        points, colors = self.pcl_extractor.extract_pointcloud(obs)
        points = torch.from_numpy(points).float().cuda()
        colors = torch.from_numpy(colors).float().cuda()
        proprio = torch.from_numpy(obs["proprio"]).float().cuda()

        clicks, ee_pos, ee_euler, gripper_cmd, mode = self.waypoint_policy.inference(
            points, colors, proprio, num_pass=1
        )

        if vis:
            click_probs = self.waypoint_policy.inference_click_probs(
                points, colors, proprio
            ).cpu()
            vis_pred(points, colors, clicks, ee_pos, ee_euler, gripper_cmd, click_probs)

        self.env.move_to(ee_pos, ee_euler, gripper_cmd, control_freq=self.control_freq, recorder=self.recorder)
        
        return mode

    def run_dense_mode(
        self,
        vis: int,
    ):
    
        cached_actions = []
        mode = ActMode.Dense.value
        t = 0
        consecutive_modes_required = 5 

        WAYPOINT_THRESH = 0.5
        TERMINATE_THRESH = 1.3

        mode_history = []

        while mode == ActMode.Dense.value:
            with common_utils.FreqGuard(self.control_freq):
                obs = self.env.observe()
                if vis:
                    images = [obs[view] for view in self.dense_dataset.camera_views]
                    stacked = np.hstack(images)
                    stacked = cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR)
                    cv2.imshow("img", stacked)
                    cv2.waitKey(1)
    
                dense_obs = self.dense_dataset.process_observation(obs)
                for k, v in dense_obs.items():
                    dense_obs[k] = v.cuda()
    
                if len(cached_actions) == 0:
                    with self.stopwatch.time("act"):
                        action_seq = self.dense_policy.act(dense_obs)
    
                    for action in action_seq.split(1, dim=0):
                        cached_actions.append(action.squeeze(0))

    
                action = cached_actions.pop(0)
                ee_pos, ee_euler, gripper_open, raw_mode = action.split([3, 3, 1, 1])

                if len(mode_history) == consecutive_modes_required:
                    if np.all(np.array(mode_history) < WAYPOINT_THRESH):
                        mode = ActMode.Waypoint.value
                    elif np.all(np.array(mode_history) > TERMINATE_THRESH):
                        mode = ActMode.Terminate.value
                    else:
                        mode = ActMode.Dense.value
                    mode_history = []

                mode_history.append(raw_mode.item())

                t += 1

                action = np.concatenate([ee_pos, ee_euler, [gripper_open.item()]]).astype(np.float32)
                self.recorder.record(ActMode.Dense, obs, action)
                self.env.apply_action(ee_pos.numpy(), ee_euler.numpy(), gripper_open.item(), is_delta=True)
    
        return mode

@dataclass
class EvalConfig:
    dense_weight: str = ""
    waypoint_weight: str = ""
    # original env and env overwrite
    env_cfg_path: str = ""
    # others
    vis: int = 1
    freq: float = 10

    @property
    def env_cfg(self):
        return pyrallis.load(FrankaEnvConfig, open(self.env_cfg_path))

def main(cfg: EvalConfig):

    dense_policy, dense_dataset, _ = load_model("%s/latest.pt"%cfg.dense_weight, "cuda", load_only_one=True)
    dense_policy.eval()

    if dense_policy.cfg.use_ddpm:
        common_utils.cprint(f"Warning: override to use ddim with step 10")
        dense_policy.cfg.use_ddpm = 0
        dense_policy.cfg.ddim.num_inference_timesteps = 10

    waypoint_policy, env_cfg = load_waypoint("%s/latest.pt"%cfg.waypoint_weight)
    waypoint_policy.eval()
    waypoint_policy.cuda()
    assert isinstance(env_cfg, FrankaEnvConfig)

    agent = InteractiveBot(cfg.freq, waypoint_policy, dense_policy, dense_dataset, env_cfg)
    for i in range(10):
        agent.reset()
        agent.run_episode(cfg.vis)

if __name__ == "__main__":
    """example command:

    python scripts/eval_sphinx.py --dense_weight exps/dense/coffee --waypoint_weight exps/waypoint/coffee --env_cfg_path envs/fr3.yaml --freq 10
    """
    import rich.traceback

    rich.traceback.install()
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)

    cfg = pyrallis.parse(config_class=EvalConfig)  # type: ignore
    main(cfg)
