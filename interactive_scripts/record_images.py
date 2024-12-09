from dataclasses import dataclass
import numpy as np
import pyrallis
import cv2
import common_utils
from minrobot.camera_env import CameraEnvConfig, CameraEnv
from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode

def record_episode(
    env: CameraEnv,
    recorder: DatasetRecorder,
    freq: float,
    max_len: int,
    stopwatch: common_utils.Stopwatch,
    show_camera: int,
):

    with stopwatch.time("reset"):
        env.reset() 

    for _ in range(max_len):
        with common_utils.FreqGuard(freq):
            with stopwatch.time("observe"):
                obs = env.observe()
                action = np.zeros(7) # Here is where you'd put the logic to get actions from your teleop interface
                recorder.record(ActMode.Dense, obs, action) # Record obs and action 

    stopwatch.summary() # Print out some timing info
    recorder.end_episode(save=True) # Save gif + demo as npz
    return

@dataclass
class RecordConfig:
    camera_cfg_path: str = ""
    save_path: str="demos"
    show_camera: int = 1
    freq: float = 10
    max_len: int = 50

    @property
    def env_cfg(self):
        return pyrallis.load(CameraEnvConfig, open(self.camera_cfg_path))


def main(cfg: RecordConfig):
    env = CameraEnv(cfg.env_cfg)

    # warm up camera for 10s
    for _ in range(int(cfg.freq) * 5):
        with common_utils.FreqGuard(cfg.freq):
            env.observe()

    stopwatch = common_utils.Stopwatch()
    recorder = DatasetRecorder(cfg.save_path)
    record_episode(env, recorder, cfg.freq, cfg.max_len, stopwatch, cfg.show_camera)

if __name__ == "__main__":
    """example command:

    python interactive_scripts/record_images.py --camera_cfg_path minrobot/cameras.yaml --max_len 50 --freq 10
    """
    cfg = pyrallis.parse(config_class=RecordConfig)  # type: ignore
    main(cfg)
