import numpy as np
import time
import cv2
import open3d as o3d
import copy
from sklearn.neighbors import NearestNeighbors


def update_point_cloud(vis, pcd, points, colors):
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


def display_point_cloud(points, colors, reference_point=False):
    # Convert points and colors to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud", width=800, height=600)
    vis.add_geometry(pcd)

    if reference_point:
        cube = o3d.geometry.TriangleMesh.create_box(width=0.025, height=0.025, depth=0.025)
        yellow_color = np.array([1.0, 1.0, 0.0])  # RGB values for yellow
        cube_transform = np.eye(4)
        cube_pos = [0.60, 0.015, 0.0125]
        cube_transform[0:3, 3] = cube_pos
        cube.paint_uniform_color(yellow_color)
        cube.transform(cube_transform)
        vis.add_geometry(cube)

    # Render the initial point cloud
    vis.poll_events()
    vis.update_renderer()

    return vis, pcd


def visualize_pointcloud(pointcloud):
    o3d.visualization.draw_geometries([pointcloud])


def deproject(depth_image, K, tf=np.eye(4), base_units=-3):
    # Convert depth image to meters
    depth_image_m = depth_image * (10**base_units)

    h, w = depth_image.shape
    i, j = np.indices((h, w))

    # Create homogeneous coordinates for pixels
    pixels_homog = np.stack([j.ravel(), i.ravel(), np.ones_like(i).ravel()], axis=0)

    # Compute the 3D points in the camera frame
    depth_arr = depth_image_m.ravel()
    points_3d = np.linalg.inv(K) @ pixels_homog * depth_arr

    # Transform the points to the target frame
    points_3d_homog = np.vstack([points_3d, np.ones(points_3d.shape[1])])
    points_3d_transf = (tf @ points_3d_homog).T[:, :3]

    return points_3d_transf


# def deproject(depth_image, K, tf = np.eye(4), base_units=-3):
#    depth_image = depth_image*(10**base_units) # convert mm to m (TODO)
#
#    h,w = depth_image.shape
#    row_indices = np.arange(h)
#    col_indices = np.arange(w)
#    pixel_grid = np.meshgrid(col_indices, row_indices)
#    pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
#
#    pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
#    depth_arr = np.tile(depth_image.flatten(), [3, 1])
#    points_3d = depth_arr * np.linalg.inv(K).dot(pixels_homog)
#
#    points_3d_transf = np.vstack((points_3d, np.ones([1,points_3d.shape[1]])))
#    points_3d_transf = ((tf.dot(points_3d_transf)).T)[:, 0:3]
#
#    return points_3d_transf


def deproject_pixels(pixels, depth_image, K, tf=np.eye(4), base_units=-3):
    h, w = depth_image.shape
    all_points = deproject(depth_image, K, tf, base_units)
    all_points = np.reshape(all_points, (h, w, 3))
    mask = np.zeros((h, w))
    mask[pixels[:, 1], pixels[:, 0]] = 1
    mask = mask.astype(bool)
    return all_points[mask]
    # idxs = pixels[:,0] + pixels[:,1]*h
    # return all_points[idxs]


def project(robot_point, K, TRC):
    xr, yr, zr = robot_point
    xc, yc, zc = TRC.dot(np.array([xr, yr, zr, 1]))
    u, v, depth = K.dot(np.array([xc, yc, zc]))
    u /= depth
    v /= depth
    px = np.array([int(u), int(v)])
    return px


def transform_points(tf, points_3d):
    points_3d = points_3d.T
    points_3d_transf = np.vstack((points_3d, np.ones([1, points_3d.shape[1]])))
    points_3d_transf = ((tf.dot(points_3d_transf)).T)[:, 0:3]
    # print(points_3d_transf.shape)
    return points_3d_transf


def crop(pcd, min_bound=[0.0, -0.35, -0.05], max_bound=[0.9, 0.4, 1.0]):
    # Use a single logical operation to compute the indices within the bounding box
    idxs = np.all((pcd > min_bound) & (pcd < max_bound), axis=1)
    return idxs


# def crop(pcd, min_bound=[0.0,-0.35,-0.05], max_bound=[0.9, 0.4, 1.0]):
#    idxs = np.logical_and(np.logical_and(
#                  np.logical_and(pcd[:,0] > min_bound[0], pcd[:,0] < max_bound[0]),
#                  np.logical_and(pcd[:,1] > min_bound[1], pcd[:,1] < max_bound[1])),
#                  np.logical_and(pcd[:,2] > min_bound[2], pcd[:,2] < max_bound[2]))
#
#    return idxs


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # print("Transformation: " + str(transformation))
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def rescale_pcd(pcd, scale=1.0):
    pcd_temp = copy.deepcopy(pcd)
    points = np.asarray(pcd.points)
    new_points = points * scale
    pcd_temp.points = o3d.utility.Vector3dVector(new_points)
    return pcd_temp


def align_pcds(pcds, visualize=False):
    target_pcd = pcds[0]

    threshold = 0.02
    trans_init = np.eye(4)
    # scale = 3.
    scale = 4.0
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=50
    )

    aligned_pcds = [target_pcd]

    target_pcd = rescale_pcd(target_pcd, scale=scale)

    for source_pcd in pcds[1:]:
        source_pcd = rescale_pcd(source_pcd, scale=scale)

        start = time.time()

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )
        end = time.time()

        # draw_registration_result(source_pcd, target_pcd, reg_p2p.transformation)
        tf = reg_p2p.transformation

        # print('TF', tf, end-start)
        # tf = trans_init

        # source_pcd = source_pcd.transform(tf)

        source_pcd_transf = copy.deepcopy(source_pcd)
        source_pcd_transf.transform(tf)
        source_pcd_transf = rescale_pcd(source_pcd_transf, 1 / scale)

        aligned_pcds.append(source_pcd_transf)

    return aligned_pcds


def merge_pcls(pcls, colors, origin=[0, 0, 0], visualize=False):
    pcds = []
    for pcl, color in zip(pcls, colors):
        # Check if pcl needs to be converted into array
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.colors = o3d.utility.Vector3dVector(color)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        pcd = pcd.select_by_index(ind)
        pcds.append(pcd)

    aligned_pcds = align_pcds(pcds, visualize=visualize)

    if visualize:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=origin)
        aligned_pcds.append(mesh_frame)
        o3d.visualization.draw_geometries(aligned_pcds)

    pcd_combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        pcd_combined += pcd

    return pcd_combined


def denoise(depth_img):
    max_val = np.amax(depth_img)
    min_val = np.amin(depth_img)
    normalized = depth_img - min_val / (max_val - min_val)
    normalized_vis = cv2.normalize(normalized, 0, 255, cv2.NORM_MINMAX)
    idxs = np.where(normalized_vis.ravel() > 0)[0]
    return idxs


def pix2pix_neighborhood(img, waypoint_proj):
    height, width, _ = img.shape

    pixels = []
    for i in range(width):
        for j in range(height):
            pixels.append([i, j])

    pixels = np.array(pixels)

    nbrs = NearestNeighbors(radius=3).fit(pixels)
    dists, idxs = nbrs.radius_neighbors(np.reshape(waypoint_proj, (-1, 2)))

    pixels = pixels[idxs[0]]
    return pixels


def point2point_neighborhood(source_points, target_points):
    nbrs = NearestNeighbors(n_neighbors=1).fit(target_points)
    dists, idxs = nbrs.kneighbors(source_points, return_distance=True)
    # thresh = 0.1
    thresh = 0.03
    idxs = np.where(dists < thresh)[0]
    return idxs
