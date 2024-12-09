import os
import open3d as o3d
import argparse
import pyrallis
import torch
import numpy as np

from dataset_utils.waypoint_dataset import PointCloudDataset
import scripts.train_waypoint
from models.waypoint_transformer import WaypointTransformer


def load(model_path):
    train_cfg_path = os.path.join(os.path.dirname(model_path), "cfg.yaml")
    train_cfg = pyrallis.load(  # type: ignore
        scripts.train_waypoint.MainConfig, open(train_cfg_path, "r")
    )
    policy = WaypointTransformer(train_cfg.waypoint)
    policy.load_state_dict(torch.load(model_path))

    return policy, train_cfg


def pred_click_heatmap(policy: WaypointTransformer, dataset: PointCloudDataset, idx):
    data = dataset.datas[idx]
    xyz = torch.from_numpy(data["xyz"]).float()
    color = torch.from_numpy(data["xyz_color"]).float()
    proprio = torch.from_numpy(data["proprio"]).float()

    with torch.no_grad():
        click_probs = policy.inference_click_probs(xyz, color, proprio)
        # fps_points = fps_points.cpu()
        # fps_colors = fps_colors.cpu()
        click_probs = click_probs.cpu()

    click_probs = click_probs / click_probs.max()
    click_probs = click_probs.unsqueeze(1)
    # color.fill_(0.7)
    color = color * (1 - click_probs) + click_probs * torch.tensor([[1.0, 0.0, 0.0]])

    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(xyz.numpy())
    vis_pcd.colors = o3d.utility.Vector3dVector(color.numpy())

    pcd_save_path = os.path.join(dataset.cfg.path, "eval_heat", f"{idx:05d}.pcd")
    if not os.path.exists(os.path.dirname(pcd_save_path)):
        os.mkdir(os.path.dirname(pcd_save_path))

    print(f"saving to {pcd_save_path}")
    o3d.io.write_point_cloud(pcd_save_path, vis_pcd)


def pred(policy: WaypointTransformer, dataset: PointCloudDataset, idx, num_pass):
    data = dataset.datas[idx]
    xyz = torch.from_numpy(data["xyz"]).float()
    color = torch.from_numpy(data["xyz_color"]).float()
    proprio = torch.from_numpy(data["proprio"]).float()

    with torch.no_grad():
        click_indices, pred_pos, pred_rot, gripper = policy.inference(
            xyz, color, proprio, num_pass=num_pass
        )

    points = xyz.numpy()
    colors = color.numpy()
    user_clicks = data["user_clicks"] # [1024], binary labels
    assert len(user_clicks.shape) == 1

    vis_pcd = o3d.geometry.PointCloud()
    big_points = []
    for i in range(len(user_clicks)):
        draw_big = False
        if user_clicks[i] > 0 and i in click_indices:
            # correct prediction: blue
            colors[i] = [0, 0, 1]
            draw_big = True
        elif user_clicks[i] > 0:
            # missed click: red
            colors[i] = [1, 0, 0]
        elif i in click_indices:
            # wrong pred: black
            colors[i] = [0, 0, 0]
            draw_big = True
        else:
            colors[i] = [0.7, 0.7, 0.7]

        if draw_big:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            sphere.paint_uniform_color(colors[i])
            sphere.translate(points[i])
            big_points.append(sphere)

    target_pos_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    target_pos_vis.paint_uniform_color([1.0, 0.5, 0.0])
    target_pos_vis.translate(data["action_pos"])
    big_points.append(target_pos_vis)

    pred_pos_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    pred_pos_vis.paint_uniform_color([1.0, 0.0, 0.5])
    pred_pos_vis.translate(pred_pos)
    big_points.append(pred_pos_vis)

    print("action diff:", 100 * (data["action_pos"] - pred_pos))
    print(f"err with off: {100 * np.sqrt(np.sum((data['action_pos'] - pred_pos)**2)):.6f}")

    t_rot = data["action_euler"] / np.pi * 180
    p_rot = pred_rot / np.pi * 180
    print("rot diff:", t_rot, p_rot)
    print("----------")

    vis_pcd.points = o3d.utility.Vector3dVector(points)
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)

    for big in big_points:
        big_pcd = big.sample_points_uniformly(number_of_points=500)
        vis_pcd = vis_pcd + big_pcd

    # o3d.visualization.draw_geometries([pcd])
    pcd_save_path = os.path.join(dataset.cfg.path, "eval", f"{idx:05d}.pcd")
    if not os.path.exists(os.path.dirname(pcd_save_path)):
        os.mkdir(os.path.dirname(pcd_save_path))

    print(f"saving to {pcd_save_path}")
    o3d.io.write_point_cloud(pcd_save_path, vis_pcd)


def main():
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="waypoint_model", help="model name")
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--heatmap", type=int, default=1)
    args = parser.parse_args()

    policy, train_cfg = load(args.model)
    policy = policy.cuda()
    policy.train(False)
    policy.cfg.topk_eval = args.topk

    train_cfg.dataset.repeat = 1
    dataset = PointCloudDataset(train_cfg.dataset, use_euler=True, npoints=1024, split="dev")

    for i in range(len(dataset)):
        if args.heatmap:
            pred_click_heatmap(policy, dataset, i)
        else:
            pred(policy, dataset, i, args.num_pass)


main()
