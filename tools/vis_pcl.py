import argparse
import os
import numpy as np
import open3d as o3d
import torch

from models.pointnet2_utils import farthest_point_sample
from dataset_utils.waypoint_dataset import augment_with_rotation, augment_with_translation


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None)
parser.add_argument("--num_sample", type=int, default=-1)
parser.add_argument("--aug", type=int, default=0)
parser.add_argument("--folder", type=str, default=None)
args = parser.parse_args()


def sample(xyz: torch.Tensor, colors: torch.Tensor, num_sample):
    indices = farthest_point_sample(xyz.unsqueeze(0), num_sample).squeeze(0)
    xyz = xyz[indices, :]
    colors = colors[indices, :]
    return xyz, colors


if args.path is not None:
    pcd = o3d.io.read_point_cloud(args.path)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)

    def change_image(vis):
        vis.clear_geometries()

        points = torch.from_numpy(np.asarray(pcd.points)).float()
        colors = torch.from_numpy(np.asarray(pcd.colors)).float()

        if args.num_sample > 0:
            points, colors = sample(points, colors, args.num_sample)

        if args.aug >= 1:
            points, colors, _, _ = augment_with_translation(
                points, colors, torch.rand(3), torch.rand(7)
            )
        if args.aug >= 2:
            points, *_ = augment_with_rotation(
                points, torch.rand(3), torch.rand(3), torch.rand(7), 0.1
            )

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points.numpy())
        new_pcd.colors = o3d.utility.Vector3dVector(colors.numpy())

        vis.clear_geometries()
        vis.add_geometry(new_pcd)
        vis.update_renderer()
        return False

    vis.register_key_callback(ord("X"), change_image)
    vis.run()

elif args.folder is not None:
    fns = list(sorted([fn for fn in os.listdir(args.folder) if "pcd" in fn]))
    fns = [os.path.join(args.folder, fn) for fn in fns]
    for fn in fns:
        print(fn)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    idx = 0
    vis.add_geometry(o3d.io.read_point_cloud(fns[idx]))

    def change_image(vis):
        global idx
        idx += 1
        vis.clear_geometries()
        vis.add_geometry(o3d.io.read_point_cloud(fns[idx]))
        vis.update_renderer()
        return False

    vis.register_key_callback(ord("A"), change_image)
    vis.run()
