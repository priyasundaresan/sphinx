from dataclasses import dataclass
import numpy as np
import zerorpc
from scipy.spatial.transform import Rotation
from envs.robot_utils import Proprio, position_action_to_delta_action


@dataclass
class ControllerConfig:
    server_ip: str = "localhost:4242"
    max_pos_delta: float = 0.05
    max_euler_delta: float = 0.2


class Controller:
    def __init__(self, cfg: ControllerConfig):
        self.rpc = zerorpc.Client(heartbeat=20)
        self.rpc.connect(cfg.server_ip)
        print("Connection established:", self.rpc.hello())

        self.home = np.pi * np.array([0, -0.25, 0, -0.75, 0, 0.5, 0], dtype=np.float32)
        self.rpc.set_home_joints(self.home.tolist())
        self.rpc.init_robot()

        self.cfg = cfg
        self.action_dim = 7

    def reset(self, randomize: bool) -> None:
        print(f"[controller] randomize? {randomize}")
        self.rpc.reset(randomize)

    def get_proprio(self) -> Proprio:
        proprio_dict: dict = self.rpc.get_proprio()  # type: ignore
        proprio = Proprio(
            eef_pos=proprio_dict["eef_pos"],
            eef_quat=proprio_dict["eef_quat"],
            gripper_open=proprio_dict["gripper_open"],
        )
        return proprio

    def delta_control(self, delta_pos: np.ndarray, delta_euler: np.ndarray, gripper_open: float):
        # (optional?) clip with max delta
        curr_proprio = self.get_proprio()
        delta_pos = np.clip(delta_pos, -self.cfg.max_pos_delta, self.cfg.max_pos_delta)
        delta_euler = np.clip(delta_euler, -self.cfg.max_euler_delta, self.cfg.max_euler_delta)

        # compute new pos and new quat
        new_pos = curr_proprio.eef_pos + delta_pos
        curr_rot = Rotation.from_euler("xyz", curr_proprio.eef_euler)
        delta_rot = Rotation.from_euler("xyz", delta_euler)
        new_quat = (delta_rot * curr_rot).as_quat()  # type: ignore

        self.rpc.update(new_pos.tolist(), new_quat.tolist(), gripper_open)
        self.curr_proprio = None

    def position_control(self, new_pos: np.ndarray, new_euler: np.ndarray, gripper_open: float):
        new_quat = Rotation.from_euler("xyz", new_euler).as_quat()  # type: ignore
        self.rpc.update(new_pos.tolist(), new_quat.tolist(), gripper_open)

    def position_to_delta(self, new_pos: np.ndarray, new_euler: np.ndarray):
        proprio = self.get_proprio()
        return position_action_to_delta_action(
            proprio.eef_pos, proprio.eef_euler, new_pos, new_euler
        )
