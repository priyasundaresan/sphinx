import json
import os
import argparse
import asyncio
import websockets
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
import msgpack
import pyrallis
import numpy as np

import common_utils
from interactive_scripts.interactive_utils import serve
from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode
from envs.robomimic_env import RobomimicEnv, RobomimicEnvConfig
# from spacemouse_utils.spacemouse import SpaceMouseInterface


class InteractiveBot:
    def __init__(
        self,
        task,
        num_point,
        stream_freq,
        data_folder,
    ):
        self.num_point = num_point
        self.stream_freq = stream_freq
        self.data_folder = data_folder

        if task == "can":
            cfg = RobomimicEnvConfig(
                name="PickPlaceCan",
                cameras=["agentview", "robot0_eye_in_hand"],
                max_len=3000,
                record_sim_state=1,
                crop_table=0,
            )
        elif task == "can_crop":
            cfg = RobomimicEnvConfig(
                name="PickPlaceCan",
                cameras=["agentview", "robot0_eye_in_hand"],
                max_len=3000,
                record_sim_state=1,
                crop_table=1,
            )
        elif task == "square":
            cfg = RobomimicEnvConfig(
                name="NutAssemblySquare",
                cameras=["agentview", "robot0_eye_in_hand", "sideview"],
                max_len=3000,
                record_sim_state=1,
                crop_table=1,
            )
        else:
            assert False, f"unknown task {task}"

        self.env = RobomimicEnv(cfg, on_screen_render=True, verbose=True)
        assert self.env.cfg.record_sim_state

        self.recorder = DatasetRecorder(self.data_folder)
        self._dump_or_check_env_cfg()

        # self.teleop_interface = SpaceMouseInterface(pos_sensitivity=7.0, rot_sensitivity=18.0)
        # self.teleop_interface.start_control()

    def _dump_or_check_env_cfg(self):
        cfg_path = os.path.join(self.data_folder, "env_cfg.yaml")
        if not os.path.exists(cfg_path):
            print(f"saving env cfg to {cfg_path}")
            pyrallis.dump(self.env.cfg, open(cfg_path, "w"))  # type: ignore
        else:
            assert common_utils.check_cfg(RobomimicEnvConfig, cfg_path, self.env.cfg), \
                f"Error: {self.data_folder} contains a different config than the current one"

    def reset(self):
        next_idx = self.recorder.get_next_idx()
        np.random.seed(next_idx + 1)
        self.env.reset(render=True)

    def transform_robotframe_to_uiframe(self, waypoints):
        waypoints = np.array(waypoints)
        waypoints += np.array([0.0, 0, -0.8])
        # waypoints += np.array([-0.4, 0, 0])
        waypoints_ui = np.zeros_like(np.array(waypoints))
        transf = R.from_euler("x", -90, degrees=True)
        waypoints_ui = transf.apply(waypoints)
        rescale_amt = 10
        waypoints_ui *= rescale_amt
        return waypoints_ui

    def transform_uiframe_to_robotframe(self, waypoints):
        waypoints_rob = waypoints.copy()
        waypoints_rob /= 10.0
        transf = R.from_euler("x", 90, degrees=True)
        waypoints_rob = transf.apply(waypoints_rob)
        waypoints_rob += np.array([0.0, 0, 0.8])
        # waypoints_rob += np.array([0.4, 0, 0])
        return waypoints_rob

    def prepare_point_cloud(self, obs, ravel=True):
        points, colors = self.env.get_point_cloud(obs)

        idxs = np.random.choice(len(points), min(len(points), self.num_point), replace=False)
        points_ui = self.transform_robotframe_to_uiframe(points[idxs])
        colors = colors[idxs]

        if ravel:
            points_ui = points_ui.ravel()
            colors = colors.ravel()

        points_ui = list(points_ui)
        colors = list(colors)
        return points_ui, colors

    def calculate_fingertip_offset(self, ee_euler: np.ndarray) -> np.ndarray:
        home_fingertip_offset = np.array([0, 0, -0.0])
        # home_fingertip_offset = np.array([0, 0, -0.145])
        ee_euler_adjustment = ee_euler.copy() - np.array([-np.pi, 0, 0])
        fingertip_offset = (
            R.from_euler("xyz", ee_euler_adjustment).as_matrix() @ home_fingertip_offset
        )
        return fingertip_offset

    def init_webcontent(self):
        fingertip_pos_ui = np.zeros(3)
        ee_euler_ui = np.zeros(3)

        fingertip_pos_code = "new THREE.Vector3(%.2f, %.2f, %.2f)" % (
            fingertip_pos_ui[0],
            fingertip_pos_ui[1],
            fingertip_pos_ui[2],
        )
        ee_euler_code = "new THREE.Euler(%.2f, %.2f, %.2f)" % (
            ee_euler_ui[0],
            ee_euler_ui[1],
            ee_euler_ui[2],
        )

        with open("interactive_scripts/interactive_utils/template_demo.html") as f:
            html_content = f.read()

        html_content = html_content % (
            1,
            self.num_point,
            fingertip_pos_code,
            ee_euler_code,
            "'franka'",  # gripper name
        )

        with open("interactive_scripts/interactive_utils/index.html", "w") as f:
            f.write(html_content)

        # Start Server
        # this starts the localhost which we visit on browser
        # hold it in self to prevent destruction
        self.webserver_proc = mp.Process(target=serve.http_server)
        self.webserver_proc.start()

    def init_ui_listen_process(self):
        ui_queue = mp.Queue(maxsize=1)

        async def listen_ui(websocket):
            async for message in websocket:
                message = json.loads(message)
                if not len(message):
                    continue

                if not ui_queue.empty():
                    print(
                        "WARNING: the ui_queue is not empty, dropping new UI command. "
                        "This should not happen"
                    )
                    continue

                data = message[-1]  # Retrieve the last waypoint in the UI
                click_ui_pos = [
                    data["click"]["x"],
                    data["click"]["y"],
                    data["click"]["z"],
                ]
                fingertip_ui_pos = [
                    data["position"]["x"],
                    data["position"]["y"],
                    data["position"]["z"],
                ]
                rotation = [
                    data["orientation"]["x"],
                    data["orientation"]["y"],
                    data["orientation"]["z"],
                ]

                info = {
                    "click_ui_pos": click_ui_pos,
                    "fingertip_ui_pos": fingertip_ui_pos,
                    "rotation": rotation,
                    "gripper_open": float(data.get("url") == "http://localhost:8080/franka.obj"),
                    "done": data["done"],
                }
                print(">>>command from ui:")
                print("\tgripper open?", info["gripper_open"])
                print("\tdone?", info["done"])
                # block=True should take no extra time as the queue should be empty
                ui_queue.put(info, block=True)

        async def listen_ui_main():
            async with websockets.serve(listen_ui, "localhost", 8766):
                await asyncio.Future()

        self.listen_process = mp.Process(target=asyncio.run, args=(listen_ui_main(),))
        self.listen_process.start()

        return ui_queue

    def init_ui_update_process(self):
        ui_update_queue = mp.Queue(maxsize=1)

        async def send_data_to_web(server):
            while True:
                to_send = ui_update_queue.get()
                obs, update_ui = to_send["obs"], to_send["update_ui"]

                points, colors = self.prepare_point_cloud(obs)

                ee_pos = obs["ee_pos"]
                ee_euler = obs["ee_euler"]
                gripper_open = int(obs["gripper_open"].item() > 0.9)

                # Transform to UI frame
                fingertip_pos = ee_pos + self.calculate_fingertip_offset(ee_euler)
                fingertip_pos_ui = (
                    self.transform_robotframe_to_uiframe(fingertip_pos.reshape(1, 3)).squeeze().tolist()
                )
                ee_euler_ui = [ee_euler[0] + np.pi, ee_euler[1], ee_euler[2]]

                data = {
                    "positions": points,
                    "colors": colors,
                    "fingertip_pos_ui": fingertip_pos_ui,
                    "ee_euler_ui": ee_euler_ui,
                    "gripper_action": [1 - gripper_open],
                    "update_ui": update_ui,
                }

                msg = msgpack.packb(data)
                try:
                    # NOTE: it takes the web-side at least 20ms to render
                    await server.send(msg)
                except websockets.ConnectionClosed as e:
                    print(f"Connection closed: {e}")
                except asyncio.TimeoutError:
                    print("Sending message timed out")
                except Exception as e:
                    print(f"An error occurred: {e}")

        async def send_data_to_web_main():
            async with websockets.serve(send_data_to_web, "localhost", 8765):
                await asyncio.Future()

        self.send_process = mp.Process(target=asyncio.run, args=(send_data_to_web_main(),))
        self.send_process.start()
        return ui_update_queue

    def apply_waypoint_mode(self, ui_cmd: dict, recorder: DatasetRecorder):
        click_pos = np.array(ui_cmd["click_ui_pos"]).astype(np.float64)
        click_pos = self.transform_uiframe_to_robotframe(click_pos.reshape(1, 3)).squeeze()

        fingertip_pos_cmd = np.array(ui_cmd["fingertip_ui_pos"])
        ee_euler_cmd = np.array(
            [
                ui_cmd["rotation"][0] - np.pi,
                -ui_cmd["rotation"][2],
                ui_cmd["rotation"][1],
            ]
        )
        ee_pos_cmd = self.transform_uiframe_to_robotframe(fingertip_pos_cmd.reshape(1, 3)).squeeze()
        ee_pos_cmd -= self.calculate_fingertip_offset(ee_euler_cmd)
        gripper_open_cmd = ui_cmd["gripper_open"]

        obs = self.env.observe()
        # if np.linalg.norm(ee_pos_cmd - obs["ee_pos"]) < 0.001 and np.linalg.norm(ee_euler_cmd - obs["ee_euler"]) < 0.001:
        #     print("Skip, probably a double click")
        #     return
        action = np.concatenate([ee_pos_cmd, ee_euler_cmd, [gripper_open_cmd]], dtype=np.float32)
        recorder.record(ActMode.Waypoint, obs, action, click_pos=click_pos)

        self.env.move_to(
            ee_pos_cmd,
            ee_euler_cmd,
            gripper_open_cmd,
            recorder=recorder,
            render=True,
        )

    def maybe_apply_dense_mode(self, send_queue: mp.Queue, recorder: DatasetRecorder, stopwatch):
        return False

        # Step env with spacemouse actions
        data = self.teleop_interface.get_controller_state()
        dpos = data["dpos"]
        drot = data["raw_drotation"]
        hold = int(data["hold"])
        gripper_open = int(1 - float(data["grasp"]))  # binary
        assert isinstance(dpos, np.ndarray) and isinstance(drot, np.ndarray)

        dense_mode_triggered = np.linalg.norm(dpos) or np.linalg.norm(drot) or hold
        if not dense_mode_triggered:
            return False

        # print("dense mode triggered")
        stopwatch.record_for_freq("dense")
        with common_utils.FreqGuard(self.control_freq):
            with stopwatch.time("dense"):
                obs = self.env.observe()
                action = np.concatenate([dpos, drot, [gripper_open]]).astype(np.float32)
                recorder.record(ActMode.Dense, obs, action=action)
                self.env.apply_action(dpos, drot, gripper_open=gripper_open)
                #vis = obs['wrist_image'].copy()
                #cv2.imshow('img', vis)
                #cv2.waitKey(1)

                # NOTE: technically the observation is 1-step off, but we re-use it to save time
                # one call of self.env.observe() takes about 55ms
                with stopwatch.time("dense.send"):
                    send_queue.put({"obs": obs, "update_ui": True})
        return True

    def record_demo(self):
        self.init_webcontent()
        ui_queue = self.init_ui_listen_process()
        send_queue = self.init_ui_update_process()

        while True:
            print(common_utils.wrap_ruler("beginning new demo, resetting"))
            self.reset()
            # self.teleop_interface.gripper_is_closed = False

            print("reset done, sending first frame")
            send_queue.put({"obs": self.env.observe(), "update_ui": True})

            print("episode start!")
            self.one_episode(ui_queue, send_queue, self.recorder)

    def one_episode(self, ui_queue: mp.Queue, send_queue: mp.Queue, recorder: DatasetRecorder):
        stopwatch = common_utils.Stopwatch()

        while True:
            # waypoint mode
            if not ui_queue.empty():
                ui_cmd = ui_queue.get()
                if ui_cmd["done"]:
                    break

                self.apply_waypoint_mode(ui_cmd, recorder)
                # we can afford to call observe again here in waypoint mode
                send_queue.put({"obs": self.env.observe(), "update_ui": True})
                continue

            # # dense mode
            # dense_applied = self.maybe_apply_dense_mode(send_queue, recorder, stopwatch)

            else:
                with stopwatch.time("stream"), common_utils.FreqGuard(self.stream_freq):
                    # streaming mode
                    with stopwatch.time("stream.observe"):
                        obs = self.env.observe()
                    with stopwatch.time("stream.send"):
                        send_queue.put({"obs": obs, "update_ui": False})

            # if stopwatch.count("stream") >= 50 or stopwatch.count("dense") >= 50:
            #     stopwatch.summary(reset=True)

        recorder.end_episode(save=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--stream_freq", type=int, default=2)
    parser.add_argument("--task", type=str, default="square")
    parser.add_argument("--data_folder", type=str, default="data/dev1")
    args = parser.parse_args()

    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    mp.set_start_method('fork')  # compatibility on mac

    common_utils.kill_process_on_port(8765)

    robot = InteractiveBot(
        args.task,
        args.num_point,
        args.stream_freq,
        args.data_folder,
    )

    # robot.reset()
    robot.record_demo()


if __name__ == "__main__":
    main()
