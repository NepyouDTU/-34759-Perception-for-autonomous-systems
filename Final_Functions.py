import numpy as np
from matplotlib import pyplot as plt
import imutils
import matplotlib.cm as cm
import ipywidgets as widgets
from IPython.display import display, clear_output
from skimage import data
import glob
import open3d as o3d
import copy
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, k_means
from sklearn import linear_model
from IPython.display import Image, display
from math import *
from scipy import ndimage, signal
import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.decomposition import PCA
import math
import time
import datetime as dt
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
import ssl
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
import cv2
from numpy.linalg import inv, pinv

### Calibration of the stereo camera
def CalibrateStereoCamera():

    # -----
    img = cv2.imread("0_calib/left/000.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    print(h, w)
    image_size = (w,h)

    # -----
    nb_vertical = 5
    nb_horizontal = 7

    image_range = 37
    def get_filename(idx: int):
        return f"0_calib/left/{idx:03d}.png", f"0_calib/right/{idx:03d}.png"

    # Prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ....,(6, 5, 0).
    objp = np.zeros((nb_horizontal * nb_vertical, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nb_vertical, 0:nb_horizontal].T.reshape(-1, 2)

    objp *= 0.080 # Size of each checkerboard is 80mm.

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space.
    imgpoints_left = []  # 2D points in image plane.
    imgpoints_right = []  # 2D points in image plane.

    skip_images = [4,6,7,14,17]

    for image_idx in range(image_range):

        if image_idx in skip_images:
            continue

        fname_left, fname_right = get_filename(image_idx)

        img_left = cv2.imread(fname_left)
        img_right = cv2.imread(fname_right)

        ret_left, corners_left = cv2.findChessboardCornersSB(
            img_left, (nb_vertical, nb_horizontal), None
        )

        ret_right, corners_right = cv2.findChessboardCornersSB(
            img_right, (nb_vertical, nb_horizontal), None
        )

        # If found, add object points, image points (after refining them).
        if not (ret_left and ret_right):
            continue

        objpoints.append(objp.copy())
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        # Draw and display the corners.
        # img_drawn = cv2.drawChessboardCorners(
        #     img_left, (nb_vertical, nb_horizontal), corners_left, ret_left
        # )

        # img_drawn = cv2.drawChessboardCorners(
        #     img_drawn, (nb_vertical, nb_horizontal), corners_right, ret_right
        # )

        # plt.title(f"Index: {image_idx}")
        # plt.imshow(img_drawn)
        # plt.show()

        # cv2.imshow("img", img_drawn)
        # cv2.waitKey(500)

    cv2.destroyAllWindows()

    rms_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None
    )

    rms_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None
    )

    rms_stereo, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left, imgpoints_right,
        K_left, dist_left,
        K_right, dist_right,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, dist_left,
        K_right, dist_right,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(K_left, dist_left, R1, P1, (w,h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K_right, dist_right, R2, P2, (w,h), cv2.CV_32FC1)

    np.savez_compressed("calibration_maps.npz",
        mapLx=mapLx,
        mapLy=mapLy,
        mapRx=mapRx,
        mapRy=mapRy,
        Q=Q,
    )

    return True

def load_and_undistort_images(image_folder, format=cv2.COLOR_BGR2RGB):
    """
    Loads and undistorts images from the specified folder using precomputed calibration maps.
    
    Parameters:
    - image_folder: Path to the folder containing 'left' and 'right' subfolders with images.
    - format: cv2.COLOR_BGR2* format to convert the images to (default is cv2.COLOR_BGR2RGB).

    Returns:
    - undistorted_images_left: List of undistorted left camera images, in chosen format.
    - undistorted_images_right: List of undistorted right camera images, in chosen format.
    - undistorted_images_left_gray: List of undistorted left camera images in grayscale.
    - undistorted_images_right_gray: List of undistorted right camera images in grayscale
    """
    calibration_maps = np.load("calibration_maps.npz")

    mapLx = calibration_maps["mapLx"]
    mapLy = calibration_maps["mapLy"]
    mapRx = calibration_maps["mapRx"]
    mapRy = calibration_maps["mapRy"]
    # Load all images from the specified folder
    image_files_left = sorted(glob.glob(f"{image_folder}/left/*.png"))
    image_files_right = sorted(glob.glob(f"{image_folder}/right/*.png"))    

    undistorted_images_left = []

    for image_file in image_files_left:
        # Read the image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: Could not read image {image_file}. Skipping.")
            continue

        # Undistort the image using the provided maps
        undistorted_img = cv2.remap(img, mapLx, mapLy, interpolation=cv2.INTER_LINEAR)
        undistorted_images_left.append(undistorted_img)

    undistorted_images_right = []
    
    for image_file in image_files_right:
        # Read the image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: Could not read image {image_file}. Skipping.")
            continue

        # Undistort the image using the provided maps
        undistorted_img = cv2.remap(img, mapRx, mapRy, interpolation=cv2.INTER_LINEAR)
        undistorted_images_right.append(undistorted_img)

    
    undistorted_images_left_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in undistorted_images_left]
    undistorted_images_right_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in undistorted_images_right]
    undistorted_images_left = [cv2.cvtColor(img, format) for img in undistorted_images_left]
    undistorted_images_right = [cv2.cvtColor(img, format) for img in undistorted_images_right]

    return undistorted_images_left, undistorted_images_right, undistorted_images_left_gray, undistorted_images_right_gray

### Disparity map functions:
def create_stereo_matchers(min_disparity=0, num_disparities=5*16, block_size=9):
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        # Start searching at disparity 0, so far points can still match.
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8*block_size**2,
        P2=32*block_size**2,
        disp12MaxDiff=1,
        # Drop a pixel when the left-right and right-left disparities differ by more than 1.
        preFilterCap=31,
        # Clip the image gradient so bright spots do not dominate the match cost.
        uniquenessRatio=20,
        # Keep a match only when it beats the second-best cost by xx percent.
        speckleWindowSize=80,
        # Remove connected disparity blobs smaller than 80 pixels.
        speckleRange=2,
        # Pixels in one blob may differ by at most 2 disparity levels.
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        # Match along horizontal, vertical, and diagonal paths.
    )
    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(2000)
    wls.setSigmaColor(0.8)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    return left_matcher, right_matcher, wls

def calculate_disparity_map(left_matcher, right_matcher, wls, left_image_gray, right_image_gray):

    disp_left = left_matcher.compute(left_image_gray, right_image_gray)
    disp_right = right_matcher.compute(right_image_gray, left_image_gray)

    # Spread reliable disparity into flat pixels, but stop at a color edge
    disparity_map = wls.filter(disp_left, left_image_gray, None, disp_right).astype(np.float32) / 16
    
    # Invalid pixels are -2048. Keep them out of the color scale.
    disparity_map[disparity_map < 0] = np.nan

    return disparity_map

### Point cloud functions:
def point_cloud_from_disparity(disparity_map, left_image, Q, min_disparity=3, ):   

    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

    mask = ((disparity_map > min_disparity) & np.isfinite(points_3D).all(axis=2))

    points = points_3D[mask]
    colors = left_image[mask] / 255.0

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd

def reconstruct_robot_trajectory(left_images, left_images_gray, right_images_gray, foreground_masks, left_matcher, right_matcher, wls, Q):
    """Return robot point clouds and centroid positions for all frames."""

    robot_pcds = []
    robot_positions = []

    frames = zip(left_images, left_images_gray, right_images_gray, foreground_masks, strict=True)

    for left_image, left_gray, right_gray, foreground_mask in frames:
        disparity_map = calculate_disparity_map(
            left_matcher, right_matcher, wls, left_gray, right_gray
        )

        # Preserve pixel coordinates while excluding the background.
        robot_disparity = disparity_map.copy()
        robot_disparity[foreground_mask <= 0] = np.nan

        robot_pcd = point_cloud_from_disparity(
            robot_disparity, left_image, Q
        )
        robot_pcds.append(robot_pcd)

        points = np.asarray(robot_pcd.points)
        center = (
            points.mean(axis=0)
            if len(points) > 0
            else np.full(3, np.nan)
        )
        robot_positions.append(center)

    return robot_pcds, np.asarray(robot_positions)

### Filter functions:
def filter_robot_positions(
    robot_positions,
    dt,
    measurement_std,
    acceleration_std,
    initial_velocity_std,
):
    measurements = np.asarray(robot_positions, dtype=float)
    filtered = np.full_like(measurements, np.nan)

    kf = cv2.KalmanFilter(6, 3, 0, cv2.CV_64F)

    # Position advances according to velocity.
    F = np.eye(6)
    F[:3, 3:] = dt * np.eye(3)
    kf.transitionMatrix = F

    # We measure position, not velocity.
    H = np.zeros((3, 6))
    H[:, :3] = np.eye(3)
    kf.measurementMatrix = H

    # Measurement uncertainty.
    R = measurement_std**2 * np.eye(3)
    kf.measurementNoiseCov = R

    # Model uncertainty: unknown acceleration during each frame interval.
    G = np.vstack([
        0.5 * dt**2 * np.eye(3),
        dt * np.eye(3),
    ])
    kf.processNoiseCov = acceleration_std**2 * (G @ G.T)

    initialized = False

    for i, position in enumerate(measurements):
        valid = np.isfinite(position).all()

        if not initialized:
            if not valid:
                continue

            kf.statePost = np.concatenate([
                position, np.zeros(3)
            ]).reshape(6, 1)

            kf.errorCovPost = np.diag(
                [measurement_std**2] * 3
                + [initial_velocity_std**2] * 3
            )

            filtered[i] = position
            initialized = True
            continue

        prediction = kf.predict()

        if valid:
            estimate = kf.correct(position.reshape(3, 1))
        else:
            estimate = prediction

        filtered[i] = estimate[:3, 0]

    return filtered

### Mask functions:
def robot_masking_MOG2(left_images, history=100, var_threshold=16):
    """
    Apply MOG2 background subtraction to a list of left images and return foreground masks.
    """
    mog = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=False
    )

    foreground_masks = []

    kernel = np.ones((3, 3), np.uint8)

    for i, img in enumerate(left_images):
        if i < 2:
            mog.apply(img)  # Learn the background
            foreground_masks.append(
                np.zeros(img.shape[:2], dtype=np.uint8)
            )
            continue

        mask = mog.apply(img)


        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel
        )

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(mask)
        )

        if num_labels > 1:
            largest = 1 + np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )
            mask = np.uint8(labels == largest) * 255

        foreground_masks.append(mask)

    return foreground_masks


def detect_robot_dense_optical(undistLg, start_frame=0, end_frame=None):
    '''
    Detects the robot in a sequence of undistorted grayscale images
    by computing optical flow and generating masks based on the magnitude of the flow.

    parameters:
    - undistLg: list; undistorted and rectified images from the left camera in a stereo camera
    - start_frame: int; positive start frame number
    - end_frame: ing; end_frame number

    returns:
    - masks: List of masks only indicating the detected robot regions in each frame.
    '''
    
    flows = []
    magnitudes = []
    shape = undistLg[0].shape
    ROIy = [325, 600]
    ROIx = [0, 1280]

    foreground_masks = []

    threshold = 3.0
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # loop through the undistorted grayscale images to compute optical flow
    if end_frame == None:
        end_frame = len(undistLg) - 1
    elif end_frame > (len(undistLg) - 1):
        print("end frame out of bounds")
        return 

    for i in range(start_frame, end_frame):
        gray1 = undistLg[i][ROIy[0]:ROIy[1], ROIx[0]:ROIx[1]]
        gray2 = undistLg[i + 1][ROIy[0]:ROIy[1], ROIx[0]:ROIx[1]]

        # get Optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            0.5, 3, 15, 3, 5, 1.5, 0
        )

        # convert flow to polar coordinates to get magnitude and angle
        mag,theta = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # flows.append(flow)
        # magnitudes.append(mag)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[ROIy[0]:ROIy[1], ROIx[0]:ROIx[1]] = (mag > threshold).astype(np.uint8) * 255

         # do opening and closing to remove noise and fill gaps in the masks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=3)

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(mask)
        )

        if num_labels > 1:
            largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
            )
            mask = np.uint8(labels == largest) * 255
        
        foreground_masks.append(mask)
    # Copy the last mask to match number of images 
    foreground_masks.append(mask)
        
    return foreground_masks

### Visualization functions:
def visualize_robot_trajectory(scene_pcd, robot_positions):
    """Show robot centroids and trajectory within the reconstructed scene."""
    positions = np.asarray(robot_positions, dtype=float)
    valid = np.isfinite(positions).all(axis=1)

    if not valid.any():
        raise ValueError("No valid robot positions to visualize.")

    # Map original frame indices to the filtered point indices.
    point_indices = np.full(len(positions), -1, dtype=int)
    point_indices[valid] = np.arange(valid.sum())

    # Connect consecutive frames only, preserving gaps in tracking.
    consecutive = np.flatnonzero(valid[:-1] & valid[1:])
    lines = np.column_stack((
        point_indices[consecutive],
        point_indices[consecutive + 1],
    ))

    trajectory_points = positions[valid]

    centroids = o3d.geometry.PointCloud()
    centroids.points = o3d.utility.Vector3dVector(trajectory_points)
    centroids.paint_uniform_color([1.0, 0.0, 0.0])

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.paint_uniform_color([1.0, 0.6, 0.0])

    o3d.visualization.draw_geometries(
        [scene_pcd, centroids, trajectory],
        window_name="Robot trajectory",
        width=1280,
        height=720,)

    
def filter_robot_positions_jerk(
    robot_positions,
    dt,
    measurement_std,
    jerk_std,
    initial_velocity_std,
    initial_acceleration_std
):
    """
    parameters:
    - robot_positions: List of 3D positions (x, y, z) of the robot over time.
    - dt: Time step between measurements.
    - measurement_std: Standard deviation of the measurement noise.
    - jerk_std: Standard deviation of the jerk (third derivative of position).
    - initial_velocity_std: Standard deviation of the initial velocity estimate.
    - initial_acceleration_std: Standard deviation of the initial acceleration estimate.

    Output:
    - filtered: List of filtered 3D positions (x, y, z) of the robot over time, after applying the Kalman filter with jerk modeling.

    """

    measurements = np.asarray(robot_positions, dtype=float)
    filtered = np.full_like(measurements, np.nan)

    I = np.eye(3)
    kf = cv2.KalmanFilter(9, 3, 0, cv2.CV_64F)

    F = np.eye(9)
    F[0:3, 3:6] = dt * I
    F[0:3, 6:9] = 0.5 * dt**2 * I
    F[3:6, 6:9] = dt * I
    kf.transitionMatrix = F

    # Measurements still contain position only: x, y, z.
    H = np.zeros((3, 9))
    H[:, :3] = I
    kf.measurementMatrix = H

    # Measurement uncertainty.
    R = measurement_std**2 * np.eye(3)
    kf.measurementNoiseCov = R  

    G = np.vstack([
        (dt**3 / 6) * I,
        (dt**2 / 2) * I,
        dt * I,
    ])

    kf.processNoiseCov = jerk_std**2 * (G @ G.T)

    initialized = False

    for i, position in enumerate(measurements):
        valid = np.isfinite(position).all()

        if not initialized:
            if not valid:
                continue

            kf.statePost = np.concatenate([
                position, np.zeros(6)
            ]).reshape(9, 1)

            kf.errorCovPost = np.diag(
                [measurement_std**2] * 3
                + [initial_velocity_std**2] * 3
                + [initial_acceleration_std**2] * 3
            )

            filtered[i] = position
            initialized = True
            continue

        prediction = kf.predict()

        if valid:
            estimate = kf.correct(position.reshape(3, 1))
        else:
            estimate = prediction

        filtered[i] = estimate[:3, 0]

    return filtered

### Pipeline function:
def pipeline(left_images,right_images,left_images_gray,right_images_gray,fps):    
### pipeline 
    # load calibration values
    calibration = np.load("calibration_maps.npz")
    Q = calibration["Q"]
    fps=30
    baseline_m = 1.0 / abs(Q[3,2]) / 1000.0

    # Construct stereo matchers
    left_matcher, right_matcher, wls = create_stereo_matchers()

    # Disparity map calculation
    disparity_map = calculate_disparity_map(left_matcher, right_matcher, wls, left_images_gray[0], right_images_gray[0])

    # Construct the scene point cloud from disparity map
    scene_pcd = point_cloud_from_disparity(disparity_map, left_images[0], Q)

    # Construct robot foreground masks.
    # robot_foreground_masks = robot_masking_MOG2(left_images)
    robot_foreground_masks = detect_robot_dense_optical(left_images_gray)

    # Reconstruct the robot and its trajectory.
    robot_pcds, robot_positions = reconstruct_robot_trajectory(
        left_images,
        left_images_gray,
        right_images_gray,
        robot_foreground_masks,
        left_matcher,
        right_matcher,
        wls,
        Q,
    )

    filtered_positions = filter_robot_positions(
        robot_positions,
        dt=1 / fps,                         # Actual frame interval in seconds
        measurement_std=0.5,
        acceleration_std=5.0,
        initial_velocity_std=10,
    )

    return scene_pcd, robot_positions, filtered_positions

if __name__ == "__main__":
    CalibrateStereoCamera()
