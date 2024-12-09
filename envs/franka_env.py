from typing import Optional
import cv2
import numpy as np

import common_utils
from envs.minrobot.controller import Controller
from envs.minrobot.camera import ParallelCameras, SequentialCameras
from envs.robot_utils import get_waypoint, get_ori
from envs.robot_utils import Proprio, WaypointReach, WaypointReachConfig, MoveErrorPlot
from envs.franka_env_config import FrankaEnvConfig
from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode


class FrankaEnv:
    def __init__(self, cfg: FrankaEnvConfig):
        self.cfg = cfg

        self.controller = Controller(cfg.controller)
        self.controller.reset(randomize=False)
        self.controller.reset(randomize=False)
        proprio = self.controller.get_proprio()
        self.home_pos = proprio.eef_pos
        self.home_euler = proprio.eef_euler

        if cfg.parallel_camera:
            self.camera = ParallelCameras(cfg.cameras)
        else:
            self.camera = SequentialCameras(cfg.cameras)


    def reset(self):
        self.move_to(self.home_pos, self.home_euler, gripper_open=1, control_freq=10)
        self.controller.reset(bool(self.cfg.randomize_init))

    def observe_camera(self):
        obs = {}
        show_images = []  # for rendering

        cam_frames = self.camera.get_frames()


        for name, frames in cam_frames.items():
            show_images.append(frames["image"])

            for k, v in frames.items():
                if self.cfg.channel_first:
                    v = v.transpose(2, 0, 1)
                obs[f"{name}_{k}"] = v

        if self.cfg.show_camera:
            image = np.hstack(show_images)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", image)
            cv2.waitKey(1)

        return obs

    def observe_proprio(self) -> Proprio:
        return self.controller.get_proprio()

    def observe(self):
        obs = self.observe_camera()

        proprio = self.controller.get_proprio()
        obs["ee_pos"] = proprio.eef_pos
        obs["ee_euler"] = proprio.eef_euler
        obs["ee_quat"] = proprio.eef_quat
        obs["gripper_open"] = np.array([proprio.gripper_open])
        obs["proprio"] = proprio.eef_pos_euler

        return obs

    def apply_action(
        self, ee_pos: np.ndarray, ee_euler: np.ndarray, gripper_open: float, is_delta=True
    ):
        if is_delta:
            self.controller.delta_control(ee_pos, ee_euler, gripper_open)
        else:
            self.controller.position_control(ee_pos, ee_euler, gripper_open)

    def move_to(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: float,
        control_freq: float,
        recorder: Optional[DatasetRecorder] = None,
        max_delta=0.01,
        plot=False,
    ):
        proprio = self.controller.get_proprio()
        curr_gripper_open = float(proprio.gripper_open > 0.95)

        # functions for positional and rotational interpolation
        gen_waypoint, num_steps = get_waypoint(proprio.eef_pos, target_pos, max_delta=max_delta)
        gen_ori = get_ori(proprio.eef_euler, target_euler, num_steps)

        if plot:
            pos_plotter = MoveErrorPlot(target_pos)
            # rot_plotter = MoveErrorPlot(target_euler)

        for i in range(1, num_steps + 1):
            next_ee_pos = gen_waypoint(i)
            next_ee_euler = gen_ori(i)

            with common_utils.FreqGuard(control_freq):
                if recorder is not None:
                    obs = self.observe()
                    delta_pos, delta_euler = self.controller.position_to_delta(
                        next_ee_pos, next_ee_euler
                    )
                    action = np.concatenate([delta_pos, delta_euler, [curr_gripper_open]])
                    action = action.astype(np.float32)
                    recorder.record(ActMode.Interpolate, obs, action)

                self.controller.position_control(next_ee_pos, next_ee_euler, curr_gripper_open)

                if plot:
                    delta_pos, delta_euler = self.controller.position_to_delta(
                        next_ee_pos, next_ee_euler
                    )
                    pp = self.observe_proprio()
                    # print("delta pos norm:", np.linalg.norm(target_pos - pp.eef_pos))
                    pos_plotter.add(pp.eef_pos, target_pos, delta_pos)

        if plot:
            pos_plotter.plot()

        assert gripper_open == 0 or gripper_open == 1
        self.update_gripper(gripper_open, control_freq, recorder)

    def update_gripper(self, gripper_open, control_freq, recorder: Optional[DatasetRecorder]):
        prev_width = self.observe_proprio().gripper_open
        if np.abs(prev_width - gripper_open) == 0:
            return

        while True:
            with common_utils.FreqGuard(control_freq):
                if recorder is not None:
                    obs = self.observe()

                self.apply_action(np.zeros(3), np.zeros(3), gripper_open)

                if recorder is not None:
                    action = np.concatenate([np.zeros(3), np.zeros(3), [gripper_open]])
                    recorder.record(ActMode.Interpolate, obs, action)

            curr_width = self.observe_proprio().gripper_open
            if np.abs(curr_width - prev_width) < 0.002:
                # exit when there is no change
                return

            prev_width = curr_width

    def move_to_acc(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: float,
        control_freq: float,
        recorder: Optional[DatasetRecorder] = None,
        plot=False,
    ):
        wpr_cfg = WaypointReachConfig()
        wpr_cfg.pos_max_norm = 0.025
        wpr_cfg.pos_threshold = 0.015
        wpr_cfg.rot_max_norm = 0.1
        wpr_cfg.rot_threshold = 0.075

        waypoint_reach = WaypointReach(
            np.ones(6),  # no scaling
            target_pos,
            target_euler,
            wpr_cfg,
        )

        if plot:
            pos_plotter = MoveErrorPlot(target_pos)
            # rot_plotter = MoveErrorPlot(target_euler)

        for i in range(50):
            proprio = self.controller.get_proprio()
            delta_pos, delta_euler, reached = waypoint_reach.step(
                proprio.eef_pos, proprio.eef_euler
            )

            if reached:
                break

            with common_utils.FreqGuard(control_freq):
                # if recorder is not None:
                #     obs = self.observe()
                #     delta_pos, delta_euler = self.controller.position_to_delta(
                #         next_ee_pos, next_ee_euler
                #     )
                #     action = np.concatenate(
                #         [delta_pos, delta_euler, [self.curr_gripper_open]]
                #     ).astype(np.float32)
                #     recorder.record(ActMode.Interpolate, obs, action)

                self.controller.delta_control(delta_pos, delta_euler, self.curr_gripper_open)

                if plot:
                    # delta_pos, delta_euler = self.controller.position_to_delta(
                    #     next_ee_pos, next_ee_euler
                    # )
                    pos_plotter.add(self.observe_proprio().eef_pos, target_pos, delta_pos)

        if plot:
            pos_plotter.plot()

        assert gripper_open == 0 or gripper_open == 1
        self.update_gripper(gripper_open, control_freq, recorder)
        self.curr_gripper_open = gripper_open

    def __del__(self):
        del self.camera


def move():
    import pyrallis

    cfg = pyrallis.parse(config_class=FrankaEnvConfig)  # type: ignore
    env = FrankaEnv(cfg)

    proprio = env.observe_proprio()
    env.move_to_acc(
        proprio.eef_pos + np.array([0.2, -0.1, -0.1]),
        proprio.eef_euler + np.array([0.2, -0.2, -0.2]),
        1,
        10,
        plot=True,
    )
    del env


def show():
    import pyrallis
    from common_utils import FreqGuard, Stopwatch

    cfg = pyrallis.parse(config_class=FrankaEnvConfig)  # type: ignore
    # cfg.show_camera = 1
    env = FrankaEnv(cfg)

    # warm up
    stopwatch = Stopwatch()
    for i in range(500):
        with FreqGuard(10):
            with stopwatch.time("observe"):
                env.observe_camera()

    stopwatch.summary()
    del env


if __name__ == "__main__":
    show()
