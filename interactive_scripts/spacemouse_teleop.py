import cv2
import imageio
import numpy as np
from spacemouse_utils.spacemouse import SpaceMouseInterface
from envs.franka_env import FrankaEnv, FrankaEnvConfig
import pyrallis

class InteractiveBot:
    def __init__(self, robot_cfg):
        self.env = FrankaEnv(robot_cfg)
        self.control_freq = 10

    def reset(self):
        proprio = self.env.observe_proprio()
        ee_pos = proprio.eef_pos
        ee_euler = proprio.eef_euler
        self.env.move_to(
            ee_pos,
            ee_euler,
            1.0,
            control_freq=self.control_freq,
            recorder=None,
        )
        # Then reset
        self.env.reset()

    def run_teleop(self):
        interface = SpaceMouseInterface(pos_sensitivity=10.0, rot_sensitivity=18.0)
        interface.start_control()

        frames = []
        while True:
            data = interface.get_controller_state()

            dpos = data["dpos"]
            drot = data["raw_drotation"]

            dpos = np.array([-dpos[1], dpos[0], dpos[2]])
            drot = np.array([-drot[1], drot[0], drot[2]])

            hold = int(data["hold"])
            gripper_open = int(1 - float(data["grasp"]))  # binary

            proprio = self.env.observe_proprio()

            obs = self.env.observe()
            vis = obs['wrist_image']
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            if np.linalg.norm(dpos) or np.linalg.norm(drot) or hold:
                self.env.apply_action(dpos, drot, gripper_open=gripper_open)
            frames.append(vis)
            cv2.imshow('vis', vis)
            cv2.waitKey(20)


        gif_path = 'tmp.gif'
        imageio.mimsave(gif_path, frames, duration=0.06, loop=0)

if __name__ == "__main__":
    config = pyrallis.parse(config_class=FrankaEnvConfig, config_path="envs/fr3.yaml")
    robot = InteractiveBot(config)

    robot.reset()
    robot.run_teleop()
