import os
from dataclasses import dataclass
import random
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

from interactive_scripts.dataset_recorder import ActMode
from models.pointnet2_utils import farthest_point_sample


def augment_with_translation(
    points: torch.Tensor,
    colors: torch.Tensor,
    action_pos: torch.Tensor,
    proprio: torch.Tensor,
):
    loc_noise = torch.zeros(3)
    loc_noise.uniform_(-0.05, 0.05)
    aug_points = points + loc_noise.unsqueeze(0)

    color_noise = torch.zeros_like(colors)
    color_noise.normal_(0, 0.03)
    aug_colors = colors + color_noise

    assert action_pos.size() == loc_noise.size()
    aug_action_pos = action_pos + loc_noise
    aug_proprio = proprio.clone()
    aug_proprio[:3] += loc_noise

    return aug_points, aug_colors, aug_action_pos, aug_proprio


def augment_with_rotation(
    points: torch.Tensor,
    action_pos: torch.Tensor,
    action_euler: torch.Tensor,
    proprio: torch.Tensor,
    rotate_scale: float,
):
    assert rotate_scale < 0.5
    random_rot = R.from_euler(
        "z", np.random.uniform(-np.pi * rotate_scale, np.pi * rotate_scale)
    ).as_matrix()
    random_rot = torch.from_numpy(random_rot).float()

    # points: [#point, 3]
    mu = points.mean(dim=0)
    aug_points = (points - mu) @ random_rot.T + mu
    aug_action_pos = (action_pos - mu) @ random_rot.T + mu

    aug_action_rot_mat = random_rot @ R.from_euler("xyz", action_euler.numpy()).as_matrix()
    aug_action_euler = torch.from_numpy(R.from_matrix(aug_action_rot_mat).as_euler("xyz")).float()

    assert proprio.size(0) == 7, "proprio must be pos(3), euler(3), gripper(1)"
    aug_ee_pos = (proprio[:3] - mu) @ random_rot.T + mu
    aug_ee_euler = R.from_matrix(
        random_rot @ R.from_euler("xyz", proprio[3:6].numpy()).as_matrix()
    ).as_euler("xyz")
    aug_ee_euler = torch.from_numpy(aug_ee_euler).float()
    aug_proprio = torch.cat((aug_ee_pos, aug_ee_euler, proprio[-1:]))
    assert aug_proprio.size() == proprio.size()

    return aug_points, aug_action_pos, aug_action_euler, aug_proprio


def _load_files(root, split, split_seed, split_percent):
    fns = list(sorted([fn for fn in os.listdir(root) if "npz" in fn]))
    fns = [os.path.join(root, fn) for fn in fns]
    split_idx = int(len(fns) * split_percent)

    if split == "dev":
        return fns[:2]
    if split == "all":
        return fns

    random.Random(split_seed).shuffle(fns)
    if split == "train":
        fns = fns[:split_idx]
    elif split == "test":
        fns = fns[split_idx:]
    else:
        assert False
    return fns


def _process_episodes(extractor, fns: list[str], radius: float, aug_interpolate: float):
    episodes = []
    datas = []
    max_num_points = 0

    #TERMINAL_WINDOW = 1
    for fn in fns:
        data = np.load(fn, allow_pickle=True)["arr_0"]

        # TODO(?): truncate if reward is available
        episode = []
        curr_waypoint = None
        curr_waypoint_step = 0
        waypoint_len = 0

        target_mode = data[0]["mode"]
        for t, step in enumerate(list(data)):
            mode = step["mode"]
            if mode == ActMode.Waypoint:
                if data[t + 1]["mode"] == ActMode.Waypoint:
                    print(f"Warninig: skip step {t} in {fn} because the next one is also waypoint")
                    continue
                assert data[t + 1]["mode"] == ActMode.Interpolate

                action = step["action"]
                curr_waypoint = {
                    "pos": action[:3],
                    "euler": action[3:6],
                    "quat": R.from_euler("xyz", action[3:6]).as_quat(),  # type: ignore
                    "gripper": action[-1],
                    "click": step["click"],  # a single vector of length 3
                }
                curr_waypoint_step = t
                waypoint_len = 0
                for k in range(t + 1, len(list(data))):
                    if data[k]["mode"] != ActMode.Interpolate:
                        target_mode = data[k]["mode"]
                        break
                    waypoint_len += 1
                assert waypoint_len > 0
                # print(f"waypoint @step: {curr_waypoint_step}, len: {waypoint_len}")

            if mode not in [ActMode.Waypoint, ActMode.Interpolate]:
                # Skip dense timesteps and non-terminal timesteps
                continue

            if mode == ActMode.Interpolate:
                assert waypoint_len > 0
                progress = (t - curr_waypoint_step) / waypoint_len
                # Keep this timestep only if we are doing temporal augmentation
                if progress > aug_interpolate:
                    continue

            assert curr_waypoint is not None
            obs = step["obs"]
            points, colors = extractor.extract_pointcloud(obs)
            proprio = step["obs"]["proprio"]

            # label clicks
            dist_to_click = np.linalg.norm(
                points - np.expand_dims(curr_waypoint["click"], axis=0), axis=1
            )
            click_idxs = dist_to_click <= radius
            user_clicks = np.zeros((len(points),)).astype(points.dtype)
            user_clicks[click_idxs] = 1.0
            assert user_clicks.sum() != 0

            processed_data = {
                # input
                "xyz": points,
                "xyz_color": colors,
                "proprio": proprio,
                # to predict
                "user_clicks": user_clicks,
                "dist_to_click": dist_to_click,
                "action_pos": curr_waypoint["pos"],
                "action_euler": curr_waypoint["euler"],
                "action_quat": curr_waypoint["quat"],
                "action_gripper": curr_waypoint["gripper"],
                "target_mode": target_mode.value,  # FIXME, this should be target mode
            }
            # print(">>>", points.shape)
            episode.append(processed_data)
            datas.append(processed_data)
            max_num_points = max(max_num_points, points.shape[0])

        episodes.append(episode)
    return datas, episodes, max_num_points


@dataclass
class PointCloudDatasetConfig:
    path: str = ""
    is_real: int = 0
    split_seed: int = 1
    split_percent: float = 0.85
    repeat: int = 1
    # data format
    radius: float = 0.05
    use_dist: int = 0
    fps: int = 0
    # augmentation
    aug_interpolate: float = 0  # create data even when the current mode is interpolate
    aug_translate: int = 0
    aug_rotate: float = 0

    def __post_init__(self):
        PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
        DATASETS = {
            "square": os.path.join(PROJECT_ROOT, "data/square"),
            "can": os.path.join(PROJECT_ROOT, "data/can"),
            "coffee": os.path.join(PROJECT_ROOT, "data/coffee"),
            "drawer": os.path.join(PROJECT_ROOT, "data/drawer"),
            "cups": os.path.join(PROJECT_ROOT, "data/cups"),
            "trainbridge": os.path.join(PROJECT_ROOT, "data/trainbridge"),
        }
        if self.path in DATASETS:
            self.path = DATASETS[self.path]


class PointCloudDataset(Dataset):
    def __init__(self, cfg: PointCloudDatasetConfig, use_euler: bool, npoints: int, split: str):
        assert split in ["train", "test", "dev", "all"]
        self.cfg = cfg
        self.use_euler = use_euler
        self.npoints = npoints
        self.split = split
        self.env_cfg_path = os.path.join(cfg.path, "env_cfg.yaml")

        self.fns = _load_files(cfg.path, split, cfg.split_seed, cfg.split_percent)
        print(f"Creating {split} dataset with {len(self.fns)} demos")

        if self.cfg.is_real:
            from envs.franka_env_config import FrankaEnvConfig
            from interactive_scripts.real_pcl_extractor import RealPclExtractor
            import pyrallis

            robot_cfg = pyrallis.load(FrankaEnvConfig, open(self.env_cfg_path, "r"))  # type: ignore
            extractor = RealPclExtractor(
                ["agent1", "agent2"], robot_cfg.calib, robot_cfg.min_bound, robot_cfg.max_bound
            )
        else:
            from envs.robomimic_env import SimPclExtractor
            extractor = SimPclExtractor(cfg.path)

        self.datas, self.episodes, self.max_num_points = _process_episodes(
            extractor,
            self.fns,
            self.cfg.radius,
            self.cfg.aug_interpolate,
        )
        print(f"Total num of data item: {len(self.datas)}, max_num_points: {self.max_num_points}")

    def __len__(self):
        return len(self.datas) * self.cfg.repeat

    def __getitem__(self, index):
        """
        return:
            point_set: np.ndarray
            colors: np.ndarray
            user_clicked_labels: np.ndarray
            action_pos: np.ndarray
            action_quat: np.ndarray
            action_gripper: float
            proprio: np.ndarray
            mode: int
        """
        if self.cfg.repeat > 1:
            index = index % len(self.datas)

        data = self.datas[index]
        xyz = torch.from_numpy(data["xyz"]).float()

        if self.cfg.fps:
            # sample points with fps
            # do fps inside the dataset so that the output is fixed to npoints
            # fps takes >100ms
            indices = farthest_point_sample(xyz.unsqueeze(0), self.npoints).squeeze(0)
        else:
            # pad every data point to the same number of points
            num_padding_point = self.max_num_points - xyz.shape[0]
            padding_indices = np.random.choice(xyz.shape[0], num_padding_point, replace=True)
            indices = torch.from_numpy(np.concatenate((np.arange(xyz.shape[0]), padding_indices)))

        xyz = xyz[indices, :]
        colors = torch.from_numpy(data["xyz_color"]).float()
        colors = colors[indices, :]

        # these are prediction targets
        user_clicked_labels = torch.from_numpy(data["user_clicks"]).long()
        user_clicked_labels = user_clicked_labels[indices]
        assert user_clicked_labels.sum() != 0

        if self.cfg.use_dist:
            dist = self.cfg.radius - torch.from_numpy(data["dist_to_click"]).float()
            dist = dist[indices]
            user_clicked_labels = user_clicked_labels * dist
            assert user_clicked_labels.min() == 0
            assert user_clicked_labels.max() <= self.cfg.radius
            user_clicked_labels /= user_clicked_labels.max()

        action_pos = torch.from_numpy(data["action_pos"]).float()
        if self.use_euler:
            action_rot = torch.from_numpy(data["action_euler"]).float()
        else:
            action_rot = torch.from_numpy(data["action_quat"]).float()
        action_gripper = torch.tensor(data["action_gripper"], dtype=torch.float32)
        proprio = torch.from_numpy(data["proprio"]).float()
        target_mode = torch.tensor(data["target_mode"]).long()

        if self.cfg.aug_translate:
            xyz, colors, action_pos, proprio = augment_with_translation(
                xyz, colors, action_pos, proprio
            )
        if self.cfg.aug_rotate:
            xyz, action_pos, action_rot, proprio = augment_with_rotation(
                xyz, action_pos, action_rot, proprio, self.cfg.aug_rotate
            )

        pcd = torch.cat((xyz, colors), 1)

        return (
            pcd,
            proprio,
            user_clicked_labels,
            action_pos,
            action_rot,
            action_gripper,
            target_mode,
        )

    def save_vis(self, save_dir, render_gripper):
        save_dir = os.path.join(self.cfg.path, save_dir)
        print(f"saving vis to {save_dir}, {self.cfg.aug_interpolate=}, {self.cfg.aug_translate=}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for i in range(len(self.datas)):
            pcd, proprio, clicks, action_pos, action_rot, _, _ = self[i]
            points, colors = pcd.split([3, 3], dim=1)

            clicks = clicks.unsqueeze(1)
            red = torch.tensor([0, 0, 1], dtype=torch.float32)
            colors = red * clicks + colors * (1 - clicks)

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points.numpy())
            point_cloud.colors = o3d.utility.Vector3dVector(colors.numpy())

            if render_gripper:
                # render target gripper
                gripper_vis = o3d.io.read_triangle_mesh(
                    "interactive_scripts/interactive_utils/franka.obj"
                )
                gripper_vis.paint_uniform_color([0.8, 0.0, 0.0])
                rotation_matrix = R.from_euler("xyz", action_rot).as_matrix()
                default_rot = R.from_euler("x", -np.pi / 2).as_matrix()
                rotation_matrix = rotation_matrix @ default_rot
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:, 3][:-1] = action_pos.numpy()
                gripper_vis.transform(transform)
                point_cloud += gripper_vis.sample_points_uniformly(number_of_points=1000)

                # render curr gripper
                gripper_vis = o3d.io.read_triangle_mesh(
                    "interactive_scripts/interactive_utils/franka.obj"
                )
                gripper_vis.paint_uniform_color([0.0, 0.0, 0.8])
                rotation_matrix = R.from_euler("xyz", proprio[3:6]).as_matrix()
                default_rot = R.from_euler("x", -np.pi / 2).as_matrix()
                rotation_matrix = rotation_matrix @ default_rot
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:, 3][:-1] = proprio[:3].numpy()
                gripper_vis.transform(transform)
                point_cloud += gripper_vis.sample_points_uniformly(number_of_points=1000)

            path = os.path.join(save_dir, f"%05d.pcd" % i)
            print(f"saving to {path}")
            o3d.io.write_point_cloud(path, point_cloud)


def main():
    cfg = PointCloudDatasetConfig(
        path="data/drawer_new2",
        is_real=1,
        aug_interpolate=0,
        aug_translate=0,
        aug_rotate=0,
        use_dist=1,
        fps=0,
    )
    dataset = PointCloudDataset(cfg, use_euler=True, npoints=1024, split="all")
    d = dataset[0]
    print("target_mode", d[-1].item())
    user_clicked_labels = d[2] / d[2].sum()
    indices = user_clicked_labels > 0
    print("#positive:", user_clicked_labels[indices].size())
    print("click labels:", user_clicked_labels[indices])
    print("click labels max:", user_clicked_labels[indices].max())
    dataset.save_vis("vis_dev", render_gripper=False)


if __name__ == "__main__":
    main()
