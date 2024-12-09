import cv2
import sys
import signal
import os
import open3d as o3d
import numpy as np
from vision_utils.pc_utils import *
from vision_utils.calib_utils import detect_calibration_marker, Solver
from scipy.spatial.transform import Rotation as R
from spacemouse_utils.spacemouse import SpaceMouseInterface
from envs.franka_env import FrankaEnv, FrankaEnvConfig
import pyrallis
import select

class InteractiveBot:
    def __init__(self, robot_cfg):
        self.env = FrankaEnv(robot_cfg)
        self.control_freq = 10

        # self.transforms = None
        if os.path.exists("calib/transforms_both.npy"):
            self.transforms = np.load("calib/transforms_both.npy", allow_pickle=True).item()
        else:
            self.transforms = None
        self.interface = None
        self.VIS_EXTRINSICS = np.array([[ 0.83452733, -0.55094297, -0.0050963, -0.49741407],
                                   [-0.04885192, -0.06477759, -0.99670324,  0.18410523],
                                   [ 0.54879652,  0.83202506, -0.08097329,  0.17228202],
                                   [ 0.,          0.,          0.,          1.        ]])

    def get_transform_from_pos_quat(self, pos, quat):
        pose_transform = np.eye(4)
        rotation_matrix = R.from_quat(quat).as_matrix()
        pose_transform[:3, :3] = rotation_matrix
        pose_transform[:, 3][:-1] = pos
        return pose_transform

    def calculate_fingertip_offset(self, ee_euler: np.ndarray) -> np.ndarray:
        home_fingertip_offset = np.array([0, 0, -0.145])
        ee_euler_adjustment = ee_euler.copy() - np.array([-np.pi, 0, 0])
        fingertip_offset = (
            R.from_euler("xyz", ee_euler_adjustment).as_matrix() @ home_fingertip_offset
        )
        return fingertip_offset

    def icp_registration_numpy(self, source_points, target_points, threshold=0.001, trans_init=np.identity(4)):
        """
        Perform ICP registration between two point clouds represented as NumPy arrays. Save the relative view-to-view transformation after alignment.
        ICP is helpful for automatically aligning two point clouds from different views, where we try to find a small relative transformation between views that achieves most point overlap.

        Args:
        - source_points (numpy.ndarray): Source point cloud as Nx3 array.
        - target_points (numpy.ndarray): Target point cloud as Mx3 array.
        - threshold (float): Distance threshold for stopping criteria of ICP.
        - trans_init (numpy.ndarray): Initial transformation matrix (4x4) for ICP.

        Returns:
        - (open3d.registration.RegistrationResult): Registration result object.
        - (numpy.ndarray): Transformed source point cloud as Nx3 array.
        """

        # Convert NumPy arrays to Open3D PointCloud
        source_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(source_points)

        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(target_points)

        # Perform ICP registration
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

        # Get the transformation matrix
        transformation_matrix = reg_p2p.transformation

        # Transform the source point cloud (convert back to numpy array)
        source_transformed = np.dot(transformation_matrix[:3, :3], source_points.T).T + transformation_matrix[:3, 3]

        return transformation_matrix, source_transformed
    
    def save_intrinsics(self, views=['agent1', 'agent2']):
        for view in views:
            if hasattr(self.env, "cameras"):
                agent_intrinsics = self.env.cameras[view].get_intrinsics()["matrix"]
            else:
                agent_intrinsics = self.env.camera.get_intrinsics(view)["matrix"]
            np.save('calib/%s_intrinsics.npy'%view, agent_intrinsics)

    def take_point_cloud(self, obs, views=['agent1', 'agent2']):
        points_list = []
        colors_list = []

        for view in views:
            rgb_frame = obs[f"{view}_image"]
            depth_frame = obs[f"{view}_depth"]
            depth_frame = depth_frame.squeeze()
            if hasattr(self.env, "cameras"):
                agent_intrinsics = self.env.cameras[view].get_intrinsics()["matrix"]
            else:
                agent_intrinsics = self.env.camera.get_intrinsics(view)["matrix"]
            tf = self.transforms[view]["tcr"]
            points = deproject(depth_frame, agent_intrinsics, tf)
            colors = rgb_frame.reshape(points.shape) / 255.0
            points_list.append(points)
            colors_list.append(colors)

        merged_points = np.vstack(points_list)
        merged_colors = np.vstack(colors_list)

        idxs = crop(merged_points, min_bound=[0.35, -0.24, 0.0], max_bound=[0.7, 0.24, 0.3])
        merged_points = merged_points[idxs]
        merged_colors = merged_colors[idxs]
        return merged_points, merged_colors

    def reset(self):
        proprio = self.env.observe_proprio()
        ee_pos = proprio.eef_pos
        ee_euler = proprio.eef_euler
        self.env.move_to(
            ee_pos,
            ee_euler,
            1.0,
            control_freq=self.control_freq,
            recorder=None,
        )
        # Then reset
        self.env.reset()

    def close_gripper(self):
        self.env.apply_action(np.zeros(3), np.zeros(3), gripper_open=0.0)
        time.sleep(0.5)

    def open_gripper(self):
        self.env.apply_action(np.zeros(3), np.zeros(3), gripper_open=1.0)
        time.sleep(0.5)

    def run_calibration(self, views=['agent1', 'agent2']):
        proprio = self.env.observe_proprio()
        ee_pos = proprio.eef_pos
        ee_euler = proprio.eef_euler

        def gen_calib_waypoints(start_pos):
            waypoints = []
            for i in np.linspace(0.1,0.25,2):
                for j in np.linspace(-0.1,0.1,2):
                    for k in np.linspace(-0.2,-0.05,3):
                        waypoints.append(start_pos + [i,j,k])
            return waypoints

        waypoints = gen_calib_waypoints(ee_pos)

        solver = Solver()

        print('Opening gripper. Place the yellow cube.')
        self.open_gripper()
        input('Ready to close gripper?')
        self.close_gripper()

        waypoints_cam = {}
        waypoints_rob = {}
        for view in views:
            waypoints_cam[view] = []
            waypoints_rob[view] = []
        transforms = {}


        input('Ready to move to calibration poses?')
        for idx, waypoint in enumerate(waypoints):
            self.env.move_to(
                waypoint,
                ee_euler,
                0.0,
                control_freq=self.control_freq,
                recorder=None,
            )
            obs = self.env.observe()
            for view in ['agent1', 'agent2']:
                rgb_frame = obs[f"{view}_image"]
                depth_frame = obs[f"{view}_depth"]
                depth_frame = depth_frame.squeeze()
                detection_result = detect_calibration_marker(rgb_frame)
                if detection_result is None:
                    continue
                vis, (u,v) = detection_result
                cv2.imshow('img', vis)
                cv2.waitKey(200)
                #use_detection = not 'n' in input('Use this point?')
                #if use_detection:
                if hasattr(self.env, "cameras"):
                    agent_intrinsics = self.env.cameras[view].get_intrinsics()["matrix"]
                else:
                    agent_intrinsics = self.env.camera.get_intrinsics(view)["matrix"]
                cam_point = deproject_pixels(np.array([[u,v]]), depth_frame.squeeze(), agent_intrinsics)[0]
                if not np.any(cam_point):
                    continue
                waypoints_cam[view].append(cam_point)
                waypoint_fingertip = waypoint + self.calculate_fingertip_offset(ee_euler.copy())
                waypoints_rob[view].append(waypoint_fingertip)
                cv2.imwrite('calib/%05d_%s.jpg'%(idx, view), vis)

        for view in views:
            trc, tcr = solver.solve_transforms(np.array(waypoints_rob[view]), np.array(waypoints_cam[view]))
            transforms[view] = {'trc':trc, 'tcr':tcr}

        self.transforms = transforms
        np.save('calib/transforms_both.npy', self.transforms)
        print('Saved tf')

    def align_automatically(self):
        # Run ICP to align the two point clouds as best as possible, then save the view-to-view transform
        for i in range(5):
            # Flush both cameras for 5 steps, TODO (may not be necessary)
            obs = self.env.observe()

        points_agent2, _ = self.take_point_cloud(obs, views=['agent2'])
        points_agent1, _ = self.take_point_cloud(obs, views=['agent1'])
        agent2_to_agent1_tf, _ = self.icp_registration_numpy(points_agent2, points_agent1)
        agent2_trc = np.eye(4)
        agent2_trc[:3,:] = self.transforms['agent2']['trc']
        self.transforms['agent2']['trc'] = (agent2_to_agent1_tf @ agent2_trc)[:3,:]
        np.save('calib/transforms_both.npy', self.transforms)
        print('Saved tf after automatic ICP alignment')

    def align_manually(self, views, mode='translation'):
        if self.interface is None:
            self.interface = SpaceMouseInterface(pos_sensitivity=2.0, rot_sensitivity=0.5)
            self.interface.start_control()
        points, colors = np.random.rand(100, 3), np.random.rand(100, 3)  # Replace with actual data
        print('*****')
        print('Running alignment for: %s with mode=%s'%(views, mode))
        print('Use the spacemouse to perform pointcloud alignment.')
        print('If xy_only=True, the "z" movement of Spacemouse will not have any effect')
        print('Press "e" then "Enter" to exit without saving.')
        print('Press "Enter" to save the transforms in the current state of alignment.')
        print('*****')

        if len(views) == 2:
            # Align the overall multi-view transform

            # Have the robot move to the middle of the workspace
            print('*****')
            print('Moving robot to middle of workspace')
            print('*****')
            proprio = self.env.observe_proprio()
            target_pos = [0.59, 0.015, 0.35]
            ee_euler = proprio.eef_euler
            self.env.move_to(
                target_pos,
                ee_euler,
                1.0,
                control_freq=self.control_freq,
                recorder=None,
            )

            # Read current proprio state
            proprio = self.env.observe_proprio()
            ee_pos = proprio.eef_pos
            ee_euler = proprio.eef_euler
            ee_quat = R.from_euler("xyz", ee_euler).as_quat()

            # Render the g.t. gripper along with the current point cloud, use this to guide alignment
            gripper_vis = o3d.io.read_triangle_mesh("vis_assets/robotiq.obj")
            gripper_vis.paint_uniform_color([0.0, 0.0, 0.0])
            pose_transform = self.get_transform_from_pos_quat(ee_pos, ee_quat)
            gripper_vis.transform(pose_transform)

            vis, pcd = display_point_cloud(points, colors)
            vis.add_geometry(gripper_vis)

        else:
            # Align views by aligning the point clouds by eyeballing (try to line up the wood texture)
            vis, pcd = display_point_cloud(points, colors)

        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        cam_params.extrinsic = np.array(self.VIS_EXTRINSICS)
        ctr.convert_from_pinhole_camera_parameters(cam_params)


        while True:
            # Read spacemouse input
            data = self.interface.get_controller_state()
            dpos = data["dpos"]
            drot = data["raw_drotation"]

            obs = self.env.observe()
            points, colors = self.take_point_cloud(obs)
            if np.linalg.norm(dpos):
                for view in views:
                    if mode == 'translation':
                        self.transforms[view]["tcr"][:,3] += np.array([dpos[1], -dpos[0], 0.2*dpos[2]])
                    else:
                        drot = [drot[1], -drot[0], 0]
                        current_rot = self.transforms[view]["tcr"][:3,:3]
                        dmat = R.from_euler("xyz", drot).as_matrix()
                        new_rot = dmat @ current_rot
                        self.transforms[view]["tcr"][:3,:3] = new_rot
            update_point_cloud(vis, pcd, points, colors)  # Assuming you have a method for updating point cloud visualization

            # Non-blocking input handling for 'Enter' key
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = input().strip()
                if line.lower() == '':
                    np.save('calib/transforms_both.npy', self.transforms)
                    print("'Enter' pressed. Saved tf after manually aligning %s..."%list(views))
                    return
                elif line.lower() == 'e':
                    print("'e' pressed. Exiting without saving...")
                    return
        print('*****')
        print('\n\n')

    def calibrate(self):
        if not os.path.exists('calib'):
            os.mkdir('calib')
        self.save_intrinsics()
        self.reset()
        self.run_calibration() # This runs calib by sending the robot to poses, registering the yellow cube in camera frame, and solving/saving the initial calib/transforms.npy
        self.reset() # This resets the robot to home position
        print('*****')
        self.align_manually(['agent2'], mode='rotation') # This allows you to tweak the agent2 tf relative to agent1
        self.align_manually(['agent2'], mode='translation') # This allows you to tweak the agent2 tf relative to agent1
        self.align_manually(['agent1', 'agent2'], mode='translation') # This allows you to tweak both agent views globally
        print('*****')
        print('Calibration done!')
        print('*****')
        print('\n\n')
        print('Ignore spacemouse warnings below! Just exit with Ctrl+C or Ctrl+Backslash')

if __name__ == "__main__":
    #config = pyrallis.parse(config_class=FrankaEnvConfig, config_path="envs/fr3.yaml")
    config = pyrallis.parse(config_class=FrankaEnvConfig, config_path="envs/fr3_coffee.yaml")
    robot = InteractiveBot(config)

    robot.calibrate()
