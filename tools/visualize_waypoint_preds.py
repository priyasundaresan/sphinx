import argparse
import torch.nn as nn
import cv2
import imageio
import matplotlib.pyplot as plt
import os
from dataset_utils.pointcloud_dataloader import PointCloudDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LinearSegmentedColormap

VIS_EXTRINSICS = [
    [0.83452733, -0.55094297, -0.0050963, -0.49741407],
    [-0.04885192, -0.06477759, -0.99670324, 0.18410523],
    [0.54879652, 0.83202506, -0.08097329, 0.17228202],
    [0.0, 0.0, 0.0, 1.0],
]
CURRENT_PCD = None


def visualize_pred(
    points,
    colors,
    pred_action_pos_offsets,
    pred_user_clicked_labels,
    pred_action_pos,
    pred_action_quat,
    pred_action_gripper,
    gt_action_pos,
    gt_action_quat,
    gt_action_gripper,
    idx,
):
    global CURRENT_PCD
    input_pcd = o3d.geometry.PointCloud()
    input_pcd.points = o3d.utility.Vector3dVector(points)
    input_pcd.colors = o3d.utility.Vector3dVector(colors)

    idxs = pred_user_clicked_labels > 0
    distances_vis = np.zeros((len(points), 3))
    distances = np.linalg.norm(pred_action_pos_offsets, axis=1)
    distances_normalized = (distances - np.amin(distances)) / (
        np.amax(distances) - np.amin(distances)
    )
    distances_vis = plt.cm.inferno_r(distances_normalized)[:, :-1]

    offset_vis_pcd = o3d.geometry.PointCloud()
    offset_vis_pcd.points = o3d.utility.Vector3dVector(points)
    offset_vis_pcd.colors = o3d.utility.Vector3dVector(distances_vis)

    if pred_action_gripper:
        pred_waypoint_vis = o3d.io.read_triangle_mesh("scripts/vis_assets/robotiq_closed.obj")
    else:
        pred_waypoint_vis = o3d.io.read_triangle_mesh("scripts/vis_assets/robotiq.obj")
    pred_waypoint_vis.paint_uniform_color([0.2, 0.2, 0.3])
    pose_transform = np.eye(4)
    rotation_matrix = R.from_quat(pred_action_quat).as_matrix()
    pose_transform[:3, :3] = rotation_matrix
    pose_transform[:, 3][:-1] = pred_action_pos
    pred_waypoint_vis.transform(pose_transform)

    if gt_action_gripper:
        gt_waypoint_vis = o3d.io.read_triangle_mesh("scripts/vis_assets/robotiq_closed.obj")
    else:
        gt_waypoint_vis = o3d.io.read_triangle_mesh("scripts/vis_assets/robotiq.obj")
    gt_waypoint_vis.paint_uniform_color([0.3, 0.5, 0.4])
    gt_pose_transform = np.eye(4)
    rotation_matrix = R.from_quat(gt_action_quat).as_matrix()
    gt_pose_transform[:3, :3] = rotation_matrix
    gt_pose_transform[:, 3][:-1] = gt_action_pos
    gt_waypoint_vis.transform(gt_pose_transform)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(input_pcd)
    vis.add_geometry(gt_waypoint_vis)
    vis.add_geometry(pred_waypoint_vis)
    ctr = vis.get_view_control()

    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = np.array(VIS_EXTRINSICS)
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    CURRENT_PCD = input_pcd

    # Define a callback function for key presses
    def toggle_geometries(vis):
        global CURRENT_PCD
        if CURRENT_PCD == input_pcd:
            vis.add_geometry(offset_vis_pcd)
            vis.remove_geometry(pred_waypoint_vis)
            vis.remove_geometry(gt_waypoint_vis)
            vis.remove_geometry(input_pcd)
            CURRENT_PCD = offset_vis_pcd
        else:
            vis.add_geometry(input_pcd)
            vis.add_geometry(pred_waypoint_vis)
            vis.add_geometry(gt_waypoint_vis)
            vis.remove_geometry(offset_vis_pcd)
            CURRENT_PCD = input_pcd
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        cam_params.extrinsic = np.array(VIS_EXTRINSICS)
        ctr.convert_from_pinhole_camera_parameters(cam_params)

    vis.register_key_callback(ord("."), toggle_geometries)

    # Capture frames
    frames = []
    for i in range(250):  # Adjust number of frames as needed
        vis.poll_events()
        vis.update_renderer()
        ctr = vis.get_view_control()
        extrinsics = ctr.convert_to_pinhole_camera_parameters().extrinsic
        image = vis.capture_screen_float_buffer(False)
        frame = (np.asarray(image) * 255).astype(np.uint8)
        frame = cv2.resize(frame, (400, 256))
        frames.append(frame)

    vis.destroy_window()
    imageio.mimsave("preds/vis%05d.gif" % idx, frames, fps=30, loop=0)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = args.log_dir
    """LOG"""
    # args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_string("PARAMETER ...")
    log_string(args)

    root = args.dset

    TEST_DATASET = PointCloudDataset(root=root, npoints=args.npoint, split="test")
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_outputs = 7  # (3 action_pos_offset + 4 quaternion)
    inp_dim = 6  # (xyz, rgb)
    num_classes = 2  # (nothing, start waypoint)

    MODEL = importlib.import_module(args.model)
    policy = MODEL.get_model(num_outputs, inp_dim, num_classes).cuda()
    checkpoint = torch.load(str(experiment_dir) + "best_model.pth")
    policy.load_state_dict(checkpoint["model_state_dict"])

    if not os.path.exists("preds"):
        os.mkdir("preds")

    ctr = 0
    with torch.no_grad():
        policy = policy.eval()
        for batch_id, (
            points,
            user_clicked_labels,
            action_pos,
            action_quat,
            action_gripper,
            proprio,
            mode,
        ) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()

            points = points.float().cuda()
            proprio = proprio.float().cuda()
            print(proprio)
            points_xyz = points.T[:3, :].T
            points_xyz = points_xyz.squeeze().cpu().numpy()
            colors_xyz = points.T[3:, :].T
            colors_xyz = colors_xyz.squeeze().cpu().numpy()
            points = points.transpose(2, 1)

            (
                pred_user_clicked_labels,
                pred_action_pos_offsets,
                pred_action_quat,
                pred_action_gripper,
                pred_mode,
            ) = policy(points, proprio)
            pred_user_clicked_labels = torch.softmax(
                pred_user_clicked_labels.contiguous().view(-1, num_classes), axis=1
            )
            pred_clicked_idxs = pred_user_clicked_labels.data.max(1)[1].cpu().numpy()
            pred_clicked_idxs = np.where(pred_clicked_idxs == 1)[0]

            # Extract predicted gripper pose for relevant points
            pred_action_pos_offsets = pred_action_pos_offsets.squeeze().detach().cpu().numpy()
            pred_action_pos = (
                points_xyz[pred_clicked_idxs] - pred_action_pos_offsets[pred_clicked_idxs]
            )
            pred_action_pos = np.mean(pred_action_pos, axis=0)

            pred_action_quat = pred_action_quat.squeeze().detach().cpu().numpy()
            pred_action_quat = pred_action_quat[pred_clicked_idxs]
            pred_action_quat = np.mean(pred_action_quat, axis=0)
            norm_action_quat = 1 / np.sqrt(np.sum(pred_action_quat * pred_action_quat))
            pred_action_quat = pred_action_quat * norm_action_quat

            pred_policy_mode_probabilities = nn.Softmax(dim=1)(pred_mode)
            _, pred_mode_labels = torch.max(pred_policy_mode_probabilities, 1)
            pred_mode = pred_mode_labels.item()

            pred_action_gripper = int(torch.round(pred_action_gripper).item())

            if args.interactive:
                visualize_pred(
                    points_xyz,
                    colors_xyz,
                    pred_action_pos_offsets,
                    pred_user_clicked_labels,
                    pred_action_pos,
                    pred_action_quat,
                    pred_action_gripper,
                    action_pos,
                    action_quat,
                    action_gripper,
                    ctr,
                )

            pred_data = {
                "points": points_xyz,
                "colors": colors_xyz,
                "pred_clicked_idxs": pred_clicked_idxs,
                "pred_action_pos": pred_action_pos,
                "pred_action_quat": pred_action_quat,
                "pred_action_gripper": pred_action_gripper,
                "pred_mode": pred_mode,
                "action_pos": action_pos.squeeze().cpu().numpy(),
                "action_quat": action_quat.squeeze().cpu().numpy(),
                "action_gripper": int(action_gripper.squeeze().cpu().numpy()),
            }
            np.save("preds/%05d.npy" % batch_id, pred_data)

            print("Pred clicks", len(pred_clicked_idxs))
            print(
                "Predicted normalized waypoint and normalized gt waypoint",
                pred_action_pos,
                action_pos[0],
            )
            print(
                "Predicted gripper and gt gripper",
                pred_action_gripper,
                int(action_gripper.squeeze().cpu().numpy()),
            )
            print("Predicted quat", pred_action_quat)
            print(R.from_quat(pred_action_quat).as_euler("xyz"))
            print("Predicted euler", pred_action_quat.round(2))
            print(R.from_quat(pred_action_quat).as_euler("xyz").round(2))
            ctr += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in testing")
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument("--npoint", type=int, default=20000, help="point Number")
    parser.add_argument(
        "--log_dir", type=str, default="checkpoints/waypoint_clothhang/", help="experiment root"
    )
    parser.add_argument("--model", type=str, default="waypoint_model", help="model name")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--dset", type=str, default="data/dset_clothhang", help="dataset name")
    args = parser.parse_args()

    main(args)
