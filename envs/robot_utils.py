from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt


class MoveErrorPlot:
    def __init__(self, target):
        self.final_target = target
        self.pos = []
        self.target = []
        self.action = []

    def add(self, pos, target, action):
        self.pos.append(pos)
        self.target.append(target)
        self.action.append(action)

    def plot(self):
        fig = plt.figure(figsize=(3 * 5, 2 * 5))
        ax = fig.subplots(2, 3, squeeze=True)

        poses = np.array(self.pos)
        targets = np.array(self.target)
        actions = np.array(self.action)
        # print(poses.shape)

        for i in range(3):
            # print(i)
            ax[0][i].plot(poses[:, i], label="actual")
            ax[0][i].plot(targets[:, i], label="desired")
            ax[1][i].plot(actions[:, i], label="action")
            ax[0][i].legend()
            ax[1][i].legend()
            ymin = min(self.final_target[i], self.pos[0][i]) - 0.1
            ymax = max(self.final_target[i], self.pos[0][i]) + 0.1
            ax[0][i].set_ylim(ymin, ymax)

        plt.show()


@dataclass
class Proprio:
    # supplied as arguments
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_open: float  # gripper_width

    # computed in __init__
    gripper_open_np: np.ndarray  # gripper_width converted to array
    eef_euler: np.ndarray  # rotation in euler
    eef_pos_euler: np.ndarray

    def __init__(
        self,
        eef_pos: list[float],
        eef_quat: list[float],
        gripper_open: float,
    ):
        self.eef_pos = np.array(eef_pos)  # , dtype=np.float32)
        self.eef_quat = np.array(eef_quat)  # , dtype=np.float32)
        self.gripper_open = gripper_open

        self.gripper_open_np = np.array([self.gripper_open])  # , dtype=np.float32)
        self.eef_euler = Rotation.from_quat(self.eef_quat).as_euler("xyz")  # .astype(np.float32)
        # TODO: rename this to include gripper
        self.eef_pos_euler = np.concatenate([self.eef_pos, self.eef_euler, self.gripper_open_np])


def position_action_to_delta_action(
    curr_pos: np.ndarray, curr_euler: np.ndarray, new_pos: np.ndarray, new_euler: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    delta_pos = new_pos - curr_pos
    curr_rot = Rotation.from_euler("xyz", curr_euler)
    target_rot = Rotation.from_euler("xyz", new_euler)
    delta_rot = target_rot * curr_rot.inv()
    delta_euler = delta_rot.as_euler("xyz")
    return delta_pos, delta_euler


# positional interpolation
def get_waypoint(start_pt, target_pt, max_delta):
    total_delta = target_pt - start_pt
    num_steps = (np.linalg.norm(total_delta) // max_delta) + 1
    remainder = np.linalg.norm(total_delta) % max_delta
    if remainder > 1e-3:
        num_steps += 1
    delta = total_delta / num_steps

    def gen_waypoint(i):
        return start_pt + delta * min(i, num_steps)

    return gen_waypoint, int(num_steps)


# rotation interpolation
def get_ori(initial_euler, final_euler, num_steps):
    diff = np.linalg.norm(final_euler - initial_euler)
    ori_chg = Rotation.from_euler("xyz", [initial_euler.copy(), final_euler.copy()], degrees=False)
    if diff < 0.02 or num_steps < 2:

        def gen_ori(i):
            return initial_euler

    else:
        slerp = Slerp([1, num_steps], ori_chg)

        def gen_ori(i):
            interp_euler = slerp(i).as_euler("xyz")
            return interp_euler

    return gen_ori


def _sgd_style_step(step_size, max_norm, delta):
    delta = step_size * delta
    delta_norm = np.linalg.norm(delta)
    delta = delta / delta_norm * min(delta_norm, max_norm)
    return delta


@dataclass
class WaypointReachConfig:
    pos_threshold: float = 0.01
    pos_step_size: float = 0.5
    pos_max_norm: float = 0.1
    rot_threshold: float = 0.02
    rot_step_size: float = 0.5
    rot_max_norm: float = 0.2


class WaypointReach:
    def __init__(
        self,
        max_delta_action: np.ndarray,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        cfg: WaypointReachConfig,
    ):
        assert max_delta_action.shape == (6,)
        self.max_delta_pos = max_delta_action[:3]
        self.max_delta_euler = max_delta_action[3:]
        self.target_pos = target_pos
        self.target_euler = target_euler
        self.cfg = cfg

    def step(self, curr_pos: np.ndarray, curr_euler: np.ndarray):
        delta_pos = self.target_pos - curr_pos
        pos_reached = np.linalg.norm(delta_pos) < self.cfg.pos_threshold
        if pos_reached:
            # print(f"Pos reached, err: {np.linalg.norm(delta_pos)}")
            delta_pos_action = np.zeros_like(curr_pos)
        else:
            # print("delta pos norm:", np.linalg.norm(delta_pos))
            delta_pos = _sgd_style_step(self.cfg.pos_step_size, self.cfg.pos_max_norm, delta_pos)
            delta_pos_action = (delta_pos / self.max_delta_pos).clip(min=-1, max=1)

        # next, process rot
        curr_rot = Rotation.from_euler("xyz", curr_euler)
        target_rot = Rotation.from_euler("xyz", self.target_euler)
        delta_euler = (target_rot * curr_rot.inv()).as_euler("xyz")

        rot_reached = np.linalg.norm(delta_euler) < self.cfg.rot_threshold
        if rot_reached:
            # print(f"Rot reached, err: {np.linalg.norm(delta_euler)}")
            delta_euler_action = np.zeros_like(delta_euler)
        else:
            # print("delta euler norm:", np.linalg.norm(delta_euler))
            delta_euler = _sgd_style_step(
                self.cfg.rot_step_size, self.cfg.rot_max_norm, delta_euler
            )
            delta_euler_action = (delta_euler / self.max_delta_euler).clip(min=-1, max=1)

        reached = rot_reached and pos_reached
        return delta_pos_action, delta_euler_action, reached

    def step_(self, curr_pos: np.ndarray, curr_euler: np.ndarray):
        delta_pos = self.target_pos - curr_pos
        delta_pos = delta_pos.clip(min=-self.max_delta_pos, max=self.max_delta_pos)

        pos_err = np.linalg.norm(delta_pos)
        pos_reached = pos_err < 0.01
        if pos_reached:
            delta_pos = np.zeros_like(delta_pos)
        else:
            delta_pos = delta_pos / self.max_delta_pos

        # next, process rot
        curr_rot = Rotation.from_euler("xyz", curr_euler)
        target_rot = Rotation.from_euler("xyz", self.target_euler)
        delta_euler = (target_rot * curr_rot.inv()).as_euler("xyz")

        if np.linalg.norm(delta_euler) < 0.02:
            delta_euler = np.zeros_like(delta_euler)
            rot_reached = True
        else:
            delta_euler = delta_euler.clip(min=-self.max_delta_euler, max=self.max_delta_euler)
            delta_euler /= self.max_delta_euler
            rot_reached = False

        reached = rot_reached and pos_reached
        return delta_pos, delta_euler, reached
