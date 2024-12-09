from dataclasses import dataclass, field
from envs.minrobot.controller import ControllerConfig


# this class is in a seperate file for the ease of loading
# without importing polymetis stuff
@dataclass
class FrankaEnvConfig:
    cameras: list[dict] = field(default_factory=list)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    randomize_init: int = 0
    show_camera: int = 0
    channel_first: int = 0
    parallel_camera: int = 0
    depth_anything: int = 0
    min_bound: list[float] = field(default_factory=list)
    max_bound: list[float] = field(default_factory=list)
    calib: str = "calibration_files/short_cams"
