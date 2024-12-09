from dataclasses import dataclass, field
import os
from typing import Optional
import cv2
import numpy as np
from envs.minrobot.camera import ParallelCameras, SequentialCameras


def normalize(image):
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    norm_image = norm_image.astype(np.uint8)
    return norm_image

@dataclass
class CameraEnvConfig:
    cameras: list[dict] = field(default_factory=list)
    parallel_camera: int = 0
    show_camera: int = 0


class CameraEnv:
    def __init__(self, cfg: CameraEnvConfig):
        self.cfg = cfg
        if cfg.parallel_camera:
            self.camera = ParallelCameras(cfg.cameras)
        else:
            self.camera = SequentialCameras(cfg.cameras)


    def reset(self):
        pass

    def observe(self):
        obs = {}
        rgb_images = []  # for rendering
        depth_images = []  # for rendering
        cam_frames = self.camera.get_frames()

        raw_depths = []
        for name, frames in cam_frames.items():
            rgb_images.append(frames["image"])
            raw_depths.append(np.repeat(frames["depth"], 3, axis=2))
            depth = normalize(frames["depth"])
            depth_images.append(np.repeat(depth[:, :, np.newaxis], 3, axis=2))

            for k, v in frames.items():
                obs[f"{name}_{k}"] = v

        if self.cfg.show_camera:
            image = np.hstack(rgb_images)
            depth = np.hstack(depth_images)
            raw_depth = np.hstack(raw_depths)

            mask = (raw_depth <= 0.0)
            depth = depth * (1 - mask) + mask * np.array([[[255,0,0]]])
            depth = depth.astype(np.uint8)

            image = image * (1 - mask) + mask * np.array([[[255,0,0]]])
            image = image.astype(np.uint8)

            show = np.vstack((image, depth))
            show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
            cv2.imshow("show", show)
            cv2.waitKey(1)

        return obs

    def dump_intrinsics(self):
        root = os.path.dirname(os.path.dirname(__file__))
        for view in ["agent1", "agent2"]:
            intrinsics = self.camera.get_intrinsics(view)
            path = os.path.join(root, "calib", f"{view}_intrinsics.npy")
            print(f"saving intrinsics to {path}")
            np.save(path, intrinsics)

    def __del__(self):
        del self.camera


def main():
    import pyrallis
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--get_intrinsics", type=int, default=0)
    args = parser.parse_args()

    cfg = pyrallis.load(CameraEnvConfig, open(args.cfg, "r"))
    cameras = CameraEnv(cfg)

    if args.get_intrinsics:
        cameras.show_camera = 0
        cameras.dump_intrinsics()
    else:
        while True:
            cameras.observe()
            time.sleep(0.1)


if __name__ == "__main__":
    main()
