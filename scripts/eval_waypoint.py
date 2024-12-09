import os
import argparse
import torch
import pyrallis
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np

import common_utils
from interactive_scripts.real_pcl_extractor import RealPclExtractor
from envs.franka_env import FrankaEnv, FrankaEnvConfig
from models.waypoint_transformer import WaypointTransformer

class InteractiveBot:
    def __init__(
        self,
        control_freq,
        waypoint_policy: WaypointTransformer,
        robot_cfg: FrankaEnvConfig,
    ):
        self.control_freq = control_freq
        self.waypoint_policy = waypoint_policy
        self.env = FrankaEnv(robot_cfg)
        self.pcl_extractor = RealPclExtractor(["agent1", "agent2"], robot_cfg.calib, robot_cfg.min_bound, robot_cfg.max_bound)

    def reset(self):
        # First open gripper
        proprio = self.env.observe_proprio()
        ee_pos = proprio.eef_pos
        ee_euler = proprio.eef_euler
        self.env.move_to(ee_pos, ee_euler, 1.0, control_freq=self.control_freq)
        # Then reset
        self.env.reset()

    def run_policy(self, vis):
        # while True:
        for step in range(3):
            obs = self.env.observe()
            points, colors = self.pcl_extractor.extract_pointcloud(obs)
            points = torch.from_numpy(points).float().cuda()
            colors = torch.from_numpy(colors).float().cuda()
            proprio = torch.from_numpy(obs["proprio"]).float().cuda()

            clicks, ee_pos, ee_euler, gripper_cmd = self.waypoint_policy.inference(
                points, colors, proprio, num_pass=1
            )

            if vis:
                click_probs = self.waypoint_policy.inference_click_probs(
                    points, colors, proprio
                ).cpu()
                vis_pred(points, colors, clicks, ee_pos, ee_euler, gripper_cmd, click_probs)

                x = input("execute? (q to quitm any other key to continue)\n")
                if x.strip() == "q":
                    return

            self.env.move_to(ee_pos, ee_euler, gripper_cmd, control_freq=self.control_freq)


def vis_pred(
    points: torch.Tensor,
    colors: torch.Tensor,
    clicks: np.ndarray,
    ee_pos: np.ndarray,
    ee_euler: np.ndarray,
    gripper: float,
    click_probs: torch.Tensor,
):
    points = points.cpu().numpy()
    colors = colors.cpu()
    click_probs = click_probs / click_probs.max()
    click_probs = click_probs.unsqueeze(1)
    colors = colors * (1 - click_probs) + click_probs * torch.tensor([[1.0, 0.0, 0.0]])
    colors = colors.numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # render click
    assert len(clicks.shape) == 1
    click_points = []
    for click_idx in clicks:
        click_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        click_vis.paint_uniform_color([0.0, 0.0, 1.0])
        click_vis.translate(points[click_idx])
        click_points.append(click_vis)

    gripper_vis = o3d.io.read_triangle_mesh(
        "interactive_scripts/interactive_utils/robotiq.obj"
    )
    gripper_vis.paint_uniform_color([0.0, 0.0, 0.5])
    rotation_matrix = R.from_euler("xyz", ee_euler).as_matrix()
    default_rot = R.from_euler("x", -np.pi / 2).as_matrix()
    rotation_matrix = rotation_matrix @ default_rot
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:, 3][:-1] = ee_pos
    gripper_vis.transform(transform)

    for cp in click_points:
        cp_pcd = cp.sample_points_uniformly(number_of_points=500)
        pcd = pcd + cp_pcd

    o3d.visualization.draw_geometries([pcd, gripper_vis])


def main():
    from scripts.train_waypoint import load_waypoint
    from envs.franka_env import FrankaEnvConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--control_freq", type=int, default=10)
    parser.add_argument("--model", type=str, default="waypoint_model")
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--vis", type=int, default=1)
    args = parser.parse_args()

    policy, env_cfg = load_waypoint(args.model)
    policy.eval()
    policy.cuda()
    assert isinstance(env_cfg, FrankaEnvConfig)

    if args.topk > 0:
        print(f"Overriding topk_eval to be {args.topk}")
        policy.cfg.topk_eval = args.topk
    else:
        print(f"Eval with original topk_eval {policy.cfg.topk_eval}")

    robot = InteractiveBot(args.control_freq, policy, env_cfg)
    robot.reset()
    robot.run_policy(args.vis)
    robot.reset()


if __name__ == "__main__":
    main()
