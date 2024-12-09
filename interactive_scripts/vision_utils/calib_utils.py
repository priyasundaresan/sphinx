import numpy as np
import time
import cv2
import scipy.linalg
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pprint
from matplotlib.tri import Triangulation


def detect_calibration_marker(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 100
    # Draw rectangles around detected red blocks
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            u, v = x + w // 2, y + h // 2
            cv2.circle(image, (u, v), 4, (255, 100, 100), -1)
            return image, (u, v)


def transform(inpt, T):
    outpt = np.matrix(np.zeros(shape=inpt.shape))
    for i in range(np.size(inpt, 0)):
        outpt[i] = transform_point(np.array(inpt[i]), T)
    return outpt


def error(m1, m2):
    assert m1.shape == m2.shape
    errors = []
    for i in range(np.size(m1, 0)):
        errors.append(np.linalg.norm(m1[i] - m2[i]))
    return np.mean(errors)


def transform_data(inpt, outpt, data_in, T, data_out=None, verbose=True):
    expected = transform(data_in, T)
    if verbose:
        print("\nTransforming {0} --> {1}".format(inpt, outpt))
        print(expected)
    if data_out is not None and verbose:
        print("Actual {0} Data".format(outpt))
        print(data_out)
        print("Associated Error: " + str(error(expected, data_out)))
    return expected


def fit_to_plane(inpt):
    X, Y, Z = inpt[:, 0], inpt[:, 1], inpt[:, 2]
    A = np.c_[X, Y, np.ones((inpt.shape[0], 1))]
    coeff, _, _, _ = scipy.linalg.lstsq(A, Z)
    z = np.asscalar(coeff[0]) * X + np.asscalar(coeff[1]) * Y + np.asscalar(coeff[2])
    result = np.vstack((X, Y, z)).T
    return result


def transform_point(pt, transform):
    npt = np.ones(4)
    npt[:3] = pt
    return np.dot(transform, npt)


def solve_for_rigid_transformation(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """
    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    outpts -= outpt_mean
    inpts -= inpt_mean
    X = inpts.T
    Y = outpts.T
    covariance = np.dot(X, Y.T)
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    R = np.dot(np.dot(V, idmatrix), U.T)
    t = outpt_mean.T - np.dot(R, inpt_mean)
    T = np.zeros((3, 4))
    T[:3, :3] = R
    T[:, 3] = t
    return T


def least_squares_plane_normal(points_3d):
    x_list = points_3d[:, 0]
    y_list = points_3d[:, 1]
    z_list = points_3d[:, 2]

    A = np.concatenate((x_list, y_list, np.ones((len(x_list), 1))), axis=1)
    plane = np.matrix(np.linalg.lstsq(A, z_list)[0]).T

    return plane


def distance_to_plane(m, point):
    A = m[0, 0]
    B = m[0, 1]
    C = -1
    D = m[0, 2]
    p0 = np.array([0, 0, D])
    p1 = np.array(point)
    n = np.array([A, B, C]) / np.linalg.norm(np.array([A, B, C]))
    return np.dot(np.absolute(p0 - p1), n)


def plot_pointclouds(point_sets, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for points in point_sets:
        x, y, z = points.T
        ax.scatter3D(x, y, z, s=50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def get_transform(inpt, outpt, data_in, data_out, verbose=True):
    T = solve_for_rigid_transformation(data_in, data_out)
    if verbose:
        print("\n{0} --> {1} Transformation Matrix".format(inpt, outpt))
        print(T)
    return T


def calibrate_cam_rob(robot_chessboard_pts, cam_chessboard_pts):
    # cam_chessboard_pts = fit_to_plane(cam_chessboard_pts)
    plot_pointclouds(
        [robot_chessboard_pts, cam_chessboard_pts]
    )  # plot robot and camera chessboard points (should be unaligned)
    TR_C = get_transform("Camera", "Robot", robot_chessboard_pts, cam_chessboard_pts)  # solve for rigid transform
    TC_R = get_transform("Robot", "Camera", cam_chessboard_pts, robot_chessboard_pts)
    np.save("calib/trc.npy", TR_C)
    np.save("calib/tcr.npy", TC_R)
    cam_transformed = transform_data("Camera", "Robot", cam_chessboard_pts, TC_R, data_out=robot_chessboard_pts)
    rob_transformed = transform_data("Robot", "Camera", robot_chessboard_pts, TR_C, data_out=cam_chessboard_pts)
    plot_pointclouds([robot_chessboard_pts, cam_transformed])
    plot_pointclouds([cam_chessboard_pts, rob_transformed])
    print(TC_R @ np.array([0, 0, 0, 1]))


class Solver:
    def __init__(self):
        pass

    def solve_transforms(self, robot_pts, cam_pts, visualize=False):
        if visualize:
            plot_pointclouds([robot_pts, cam_pts])  # plot robot and camera chessboard points (should be unaligned)
        TR_C = get_transform("Camera", "Robot", robot_pts, cam_pts)  # solve for rigid transform
        TC_R = get_transform("Robot", "Camera", cam_pts, robot_pts)
        # np.save('calib/trc.npy', TR_C)
        # np.save('calib/tcr.npy', TC_R)
        cam_transformed = transform_data("Camera", "Robot", cam_pts, TC_R, data_out=robot_pts)
        rob_transformed = transform_data("Robot", "Camera", robot_pts, TR_C, data_out=cam_pts)
        if visualize:
            plot_pointclouds([robot_pts, cam_transformed])
            plot_pointclouds([cam_pts, rob_transformed])
        return TR_C, TC_R


if __name__ == "__main__":
    robot_pts = np.load("../calib/robot_pts.npy", allow_pickle=True)
    cam_pts = np.load("../calib/cam_pts.npy", allow_pickle=True).item()
    cam_pts = np.array(cam_pts[list(cam_pts.keys())[0]])
    print(robot_pts.shape, cam_pts.shape)
    solver = Solver()
    solver.solve_transforms(robot_pts, cam_pts)
