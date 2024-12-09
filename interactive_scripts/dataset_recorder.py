import numpy as np
import cv2
from enum import Enum
import imageio
import os
import glob
from typing import Optional


class ActMode(Enum):
    Waypoint = 0
    Dense = 1
    Terminate = 2
    Interpolate = 3

class DatasetRecorder:
    def __init__(self, data_folder, vis_dim=(320, 240)):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        self.vis_dim = vis_dim

        self._reset()

    def _reset(self):
        self.episode = []
        self.images = []
        self.waypoint_idx = -1

    def get_next_idx(self):
        existing_demos = glob.glob(os.path.join(self.data_folder, "demo*.npz"))
        if len(existing_demos) == 0:
            next_idx = 0
        else:
            existing_indices = [
                int(fname.split("/")[-1].split(".")[0][len("demo") :])
                for fname in existing_demos
            ]
            next_idx = np.max(existing_indices) + 1
        return next_idx

    def record(
        self,
        mode: ActMode,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        click_pos: Optional[np.ndarray] = None,
        reward: Optional[float] = None
    ):
        """mode: delta, position, waypoint"""
        if mode == ActMode.Waypoint:
            print("Recording Click:", action)
            self.waypoint_idx += 1
            waypoint_idx = self.waypoint_idx
        elif mode == ActMode.Dense:
            # print("Recording Dense:", action)
            waypoint_idx = -1
        elif mode == ActMode.Interpolate:
            print("Recording Interpolate:", action)
            waypoint_idx = self.waypoint_idx

        data = {
            "obs": obs,
            "action": action,
            "mode": mode,
            "waypoint_idx": waypoint_idx,
            "click": click_pos,
        }
        if reward is not None:
            data["reward"] = reward
        self.episode.append(data)

        views = []
        for k, v in obs.items():
            # TODO: what is this for?
            if ("image" in k or "wrist" in k) and v.ndim == 3:
                views.append(cv2.resize(v, self.vis_dim))
        self.images.append(np.hstack(views))

    def end_episode(self, save):
        if save and len(self.episode) > 0:
            next_idx = self.get_next_idx()
            mp4_path = os.path.join(self.data_folder, f"demo%05d.mp4" % next_idx)
            demo_path = os.path.join(self.data_folder, f"demo%05d.npz" % next_idx)
            print(f"saving to {mp4_path}")

            H,W,C = self.images[0].shape
            out = cv2.VideoWriter(
                mp4_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (W,H),
            )
            for i in range(len(self.images)):
                vis = self.images[i]
                if self.episode[i]["mode"] == ActMode.Dense:
                    vis[:10, :, :] = (0, 255, 0)
                self.images[i] = vis
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                out.write(vis)
                cv2.imshow("vis", vis)
                cv2.waitKey(20)

            out.release()

            np.savez_compressed(demo_path, self.episode)
        else:
            print("Episode discarded")

        self._reset()
