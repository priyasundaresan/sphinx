import argparse
import time
import numpy as np
from controller import Controller, ControllerConfig


def goto(target_x, target_y, target_z, delta, max_step):
    cfg = ControllerConfig(server_ip="tcp://172.16.0.1:4242")
    client = Controller(cfg)
    client.reset(False)

    proprio = client.get_proprio()
    x, y, z = proprio.eef_pos

    for _ in range(max_step):
        proprio = client.get_proprio()
        x, y, z = proprio.eef_pos
        print(f"{x:3f}, {y:.3f}, {z:.3f}")

        if abs(target_x - x) < 0.02 and abs(target_y - y) < 0.02 and abs(target_z - z) < 0.02:
            break

        delta_pos = np.array(
            [
                np.sign(target_x - x) * delta,
                np.sign(target_y - y) * delta,
                np.sign(target_z - z) * delta,
            ]
        )
        delta_euler = np.array([0, 0, 0])
        gripper_open = 0.5

        client.delta_control(np.array(delta_pos), np.array(delta_euler), gripper_open)
        time.sleep(0.1)

    return


"""
action space meaning:
x: + -> move away from the base
y: + -> move right (see from the desk), i.e. move left (see from the back of the base)
z: + -> move upward
rot_x: rotate along x-axis, + -> clockwise see from the back of the base
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # NOTE: 0.01 means 1 means 1 cm in real world
    parser.add_argument("--x", type=float, default=0.307)
    parser.add_argument("--y", type=float, default=0)
    parser.add_argument("--z", type=float, default=0.586)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--max_step", type=int, default=100)

    args = parser.parse_args()
    goto(args.x, args.y, args.z, args.delta, args.max_step)
