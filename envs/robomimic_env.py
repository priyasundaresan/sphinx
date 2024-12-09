import os
from dataclasses import dataclass, field
from typing import Optional
import pprint
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrallis

import robosuite
from robosuite.utils import camera_utils
from interactive_scripts.vision_utils.pc_utils import deproject
from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode
from envs.robot_utils import Proprio, WaypointReach, WaypointReachConfig, MoveErrorPlot


@dataclass
class RobomimicEnvConfig:
    name: str
    max_len: int
    cameras: list[str]
    idle_step: int = 10
    image_size: int = 224
    robots: list[str] = field(default_factory=lambda: ["Panda"])
    gripper_max_width: float = 0.08
    waytpoint_max_num_step: int = 50
    record_sim_state: int = 0
    waypoint_reach: WaypointReachConfig = field(default_factory=WaypointReachConfig)
    crop_table: int = 1


class RobomimicEnv:
    def __init__(
        self,
        cfg: RobomimicEnvConfig,
        on_screen_render=False,
        verbose=False,
    ):
        assert isinstance(cfg.cameras, list)
        self.ctrl_config = robosuite.load_controller_config(default_controller="OSC_POSE")
        assert self.ctrl_config["control_delta"]

        self.cfg = cfg
        self.verbose = verbose
        self.env = robosuite.make(
            env_name=cfg.name,
            robots=cfg.robots,
            controller_configs=self.ctrl_config,
            has_renderer=on_screen_render,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            reward_shaping=False,
            camera_names=cfg.cameras,
            camera_heights=cfg.image_size,
            camera_widths=cfg.image_size,
            camera_depths=True,
            horizon=cfg.max_len + cfg.idle_step,
            render_camera="agentview",
        )
        self.action_dim: int = self.env.action_dim

        if verbose:
            pprint.pprint(self.ctrl_config)
            print("control_freq:", self.env.control_freq)
            print("action dim:", self.action_dim)

        # bookkeeping
        self.obs = {}
        self.curr_gripper_open = 1
        self.reward = -1
        self.terminal = False
        self.num_step = 0

    def reset(self, render: bool = False):
        self.obs = self.env.reset()

        # run some idle steps for everything to be static
        for i in range(self.cfg.idle_step):
            self.apply_action(np.zeros(3), np.zeros(3), 1)
            if render:
                self.env.render()

        if self.verbose:
            print(">>>reset done: gripper open:", self.observe()["gripper_open"])

        self.curr_gripper_open = 1
        self.reward = -1
        self.terminal = False
        self.num_step = 0

    def reset_to_sim_state(self, sim_state: np.ndarray):
        self.reset()
        self.env.sim.set_state_from_flattened(sim_state)
        self.env.sim.forward()
        self.obs = self.env._get_observations(force_update=True)

    def get_camera_intrinsics(self, camera: str) -> np.ndarray:
        matrix = camera_utils.get_camera_intrinsic_matrix(
            self.env.sim, camera, self.cfg.image_size, self.cfg.image_size
        )
        return matrix

    def get_camera_extrinsics(self, camera: str) -> np.ndarray:
        matrix = camera_utils.get_camera_extrinsic_matrix(self.env.sim, camera)
        return matrix

    def observe_camera(self, show_camera=False, channel_first=False) -> dict[str, np.ndarray]:
        obs = {}
        # TODO: render it in the website instead :)
        show_images = []  # for rendering

        for name in self.cfg.cameras:
            for key in [f"{name}_image", f"{name}_depth"]:
                image: np.ndarray = self.obs[key]
                image = image[::-1]  # flip because the default images are up-side-down

                if show_camera and "depth" not in key:
                    show_images.append(image)

                if channel_first:
                    image = image.transpose(2, 0, 1)
                obs[key] = image

        if show_camera:
            image = np.hstack(show_images)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", image)
            cv2.waitKey(0)

        return obs

    def observe_proprio(self) -> Proprio:
        gripper_width = self.obs["robot0_gripper_qpos"][0] - self.obs["robot0_gripper_qpos"][1]
        return Proprio(
            eef_pos=self.obs["robot0_eef_pos"],
            eef_quat=self.obs["robot0_eef_quat"],
            gripper_open=gripper_width / self.cfg.gripper_max_width,
        )

    def observe(self) -> dict[str, np.ndarray]:
        obs = self.observe_camera()

        proprio = self.observe_proprio()
        obs["ee_pos"] = proprio.eef_pos
        obs["ee_euler"] = proprio.eef_euler
        obs["ee_quat"] = proprio.eef_quat
        obs["gripper_open"] = proprio.gripper_open_np
        obs["proprio"] = proprio.eef_pos_euler

        if self.cfg.record_sim_state:
            # obs["model"] = self.env.sim.model.get_xml()
            obs["sim_state"] = self.env.sim.get_state().flatten()

        return obs

    def apply_action(
        self, ee_pos: np.ndarray, ee_euler: np.ndarray, gripper_open: float, is_delta=True
    ):
        assert is_delta, "positional not implemented yet"
        # gripper_open = 1 (open) -> gripper_action: -1
        # gripper_open = 0 (closed) -> gripper_action: 1
        gripper_action = 1 - 2 * gripper_open
        action = np.concatenate([ee_pos, ee_euler, [gripper_action]]).astype(np.float32)
        self.obs, self.reward, self.terminal, _ = self.env.step(action)
        self.num_step += 1

        if self.verbose:
            print(f"env: {self.num_step}/{self.cfg.max_len} step, reward: {self.reward}")

    def move_to(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: float,
        recorder: Optional[DatasetRecorder] = None,
        render: bool = False,
        plot: bool = False,
    ):
        waypoint_reach = WaypointReach(
            np.array(self.ctrl_config["output_max"]),
            target_pos,
            target_euler,
            self.cfg.waypoint_reach,
        )

        proprio = self.observe_proprio()

        if plot:
            pos_plotter = MoveErrorPlot(target_pos)
            rot_plotter = MoveErrorPlot(target_euler)

        for _ in range(self.cfg.waytpoint_max_num_step):
            proprio = self.observe_proprio()
            delta_pos, delta_euler, reached = waypoint_reach.step(
                proprio.eef_pos, proprio.eef_euler
            )

            if reached:
                break

            if recorder is not None:
                obs_for_record = self.observe()

            self.apply_action(delta_pos, delta_euler, self.curr_gripper_open)

            if recorder is not None:
                action = np.concatenate([delta_pos, delta_euler, [self.curr_gripper_open]])
                recorder.record(ActMode.Interpolate, obs_for_record, action, reward=self.reward)

            if plot:
                pos_plotter.add(self.observe_proprio().eef_pos, target_pos, delta_pos)
                rot_plotter.add(self.observe_proprio().eef_euler, target_euler, delta_euler)

            if render:
                self.env.render()

            if self.terminal:
                return

        if plot:
            pos_plotter.plot()
            rot_plotter.plot()

        assert gripper_open == 0 or gripper_open == 1
        if self.curr_gripper_open != gripper_open:
            self.update_gripper(gripper_open, recorder, render)
            self.curr_gripper_open = gripper_open

    def update_gripper(
        self, gripper_open, recorder: Optional[DatasetRecorder], render: bool = False
    ):
        if self.verbose:
            print(f"[gripper] from {self.curr_gripper_open:.1f} to {gripper_open:.1f}")

        stable_count = 0
        prev_width = self.observe_proprio().gripper_open
        while True:
            if recorder is not None:
                obs_for_record = self.observe()

            self.apply_action(np.zeros(3), np.zeros(3), gripper_open)

            if recorder is not None:
                action = np.concatenate([np.zeros(3), np.zeros(3), [gripper_open]])
                recorder.record(ActMode.Interpolate, obs_for_record, action, reward=self.reward)

            curr_width = self.observe_proprio().gripper_open
            if gripper_open == 1 and np.abs(curr_width - 1) < 0.01:
                return
            else:
                if np.abs(curr_width - prev_width) < 0.002:
                    stable_count += 1
                    if stable_count >= 3:
                        # exit when there is no change
                        return
                else:
                    stable_count = 0

            prev_width = curr_width

            if render:
                self.env.render()

            if self.terminal:
                return

    def get_point_cloud(self, obs, crop_table=False):
        points_list = []
        colors_list = []
        for view in [cam for cam in self.cfg.cameras if "eye_in_hand" not in cam]:
            rgb_frame = obs[f"{view}_image"]
            depth_frame = obs[f"{view}_depth"]
            depth_frame = camera_utils.get_real_depth_map(self.env.sim, depth_frame)

            agent_intrinsics = self.get_camera_intrinsics(view)
            camera_extrinsics = self.get_camera_extrinsics(view)
            points = deproject(
                depth_frame.squeeze(), agent_intrinsics, camera_extrinsics, base_units=0
            )

            colors = rgb_frame.reshape(points.shape) / 255.0

            x_min, x_max = -0.3, 0.4
            y_min, y_max = -0.5, 0.5
            if crop_table:
                z_min, z_max = 0.83, 1.1
            else:
                z_min, z_max = 0.75, 1.1

            valid = (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
                & (points[:, 2] >= z_min)
                & (points[:, 2] <= z_max)
            )
            points = points[valid]
            colors = colors[valid]

            points_list.append(points)
            colors_list.append(colors)

        merged_points = np.vstack(points_list)
        merged_colors = np.vstack(colors_list)
        return merged_points, merged_colors

    def render_random_episode(self):
        self.reset()
        self.env.render()
        for i in range(self.cfg.max_len):
            pos_action = np.zeros(3)
            euler_action = np.zeros(3)
            self.apply_action(pos_action, euler_action, 1)

            prop = self.observe_proprio()
            print("prop:", prop.gripper_open)
            obs = self.observe()
            for k, v in obs.items():
                print(k, v.shape, v.dtype)

            self.env.render()

            if self.terminal:
                break


class SimPclExtractor:
    def __init__(self, demo_dir):
        env_cfg_path = os.path.join(demo_dir, "env_cfg.yaml")
        assert os.path.exists(env_cfg_path), f"cannot locate env config {env_cfg_path}"

        self.env_cfg = pyrallis.load(RobomimicEnvConfig, open(env_cfg_path, "r"))  # type: ignore
        self.env = RobomimicEnv(self.env_cfg)

    def extract_pointcloud(self, obs) -> tuple[np.ndarray, np.ndarray]:
        return self.env.get_point_cloud(obs, crop_table=bool(self.env_cfg.crop_table))


# Render the point cloud using Matplotlib
def render_point_cloud(points, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ## I think the XYZ axes in matplotlib vs. robosuite are diff, this makes it "top-down" in matplotlib
    # vis_points = (R.from_euler('y', np.pi/2).as_matrix() @ (points.T)).T
    vis_points = points
    ax.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], c=colors, s=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def render_pc():
    cfg = RobomimicEnvConfig(
        name="NutAssemblySquare",
        cameras=["agentview", "robot0_eye_in_hand", "sideview"],
        max_len=300,
    )
    env = RobomimicEnv(cfg, on_screen_render=True)
    env.reset()
    points, colors = env.get_point_cloud(env.observe())
    render_point_cloud(points, colors)


def main():
    cfg = RobomimicEnvConfig(
        name="NutAssemblySquare",
        cameras=["agentview", "robot0_eye_in_hand"],
        max_len=300,
    )
    env = RobomimicEnv(cfg, on_screen_render=True)
    env.reset()

    for i in range(10):
        before = env.observe_proprio()
        env.apply_action(np.array([1, 0, 0]), np.zeros(3), 1)
        after = env.observe_proprio()
        print("move before to apply some speed", i, after.eef_pos - before.eef_pos)

    proprio = env.observe_proprio()
    env.move_to(
        proprio.eef_pos,
        proprio.eef_euler + np.array([0.2, 0.0, 0.0]),
        1,
        render=True,
        plot=True,
    )
    return


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    # main()
    render_pc()
