import numpy as np
from interactive_scripts.vision_utils.pc_utils import deproject, crop


class RealPclExtractor:
    def __init__(self, views, calib_dir, min_bound, max_bound):
        self.transforms = np.load("%s/transforms_both.npy"%calib_dir, allow_pickle=True).item()
        self.intrinsics = {}
        for view in views:
            try:
                intrinsics = np.load(f"%s/{view}_intrinsics.npy"%calib_dir, allow_pickle=True).item()["matrix"]
            except:
                intrinsics = np.load(f"%s/{view}_intrinsics.npy"%calib_dir, allow_pickle=True)
            self.intrinsics[view] = intrinsics

        self.min_bound = min_bound
        self.max_bound = max_bound

    def check_intrinsics(self, camera):
        for view, loaded_intrinsics in self.intrinsics.items():
            curr_intrisics = camera.get_intrinsics(view)["matrix"]
            assert np.all(curr_intrisics == loaded_intrinsics)

    def extract_pointcloud(self, obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        points_list = []
        colors_list = []

        for view, agent_intrinsics in self.intrinsics.items():
            tf = self.transforms[view]["tcr"]

            rgb_frame = obs[f"{view}_image"]
            depth_frame = obs[f"{view}_depth"].squeeze()
            points = deproject(depth_frame, agent_intrinsics, tf)
            colors = rgb_frame.reshape(points.shape) / 255.0

            valid_depth = depth_frame.reshape(points.shape[:1]) > 0
            points = points[valid_depth]
            colors = colors[valid_depth]
            points_list.append(points)
            colors_list.append(colors)

        merged_points = np.vstack(points_list)
        merged_colors = np.vstack(colors_list)

        idxs = crop(merged_points, min_bound=self.min_bound, max_bound=self.max_bound)
        merged_points = merged_points[idxs]
        merged_colors = merged_colors[idxs]
        return merged_points, merged_colors
