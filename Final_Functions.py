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

def CalibrateStereoCamera():

    # -----
    img = cv2.imread("0_calib/left/000.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    print(h, w)
    image_size = (720, 1280)

    # -----
    nb_vertical = 5
    nb_horizontal = 7

    image_range = 37
    def get_filename(idx: int):
        return f"0_calib/left/{idx:03d}.png", f"0_calib/right/{idx:03d}.png"

    # Prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ....,(6, 5, 0).
    objp = np.zeros((nb_horizontal * nb_vertical, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nb_vertical, 0:nb_horizontal].T.reshape(-1, 2)

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
    )

    return True

def load_and_undistort_images(image_folder):
    """
    Loads and undistorts images from the specified folder using precomputed calibration maps.
    
    """
    calibration_maps = globals().update(np.load("calibration_maps.npz", allow_pickle=True))
    # Load all images from the specified folder
    image_files_left = glob.glob(f"{image_folder}/left/*.png")  
    image_files_right = glob.glob(f"{image_folder}/right/*.png")  

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

    return undistorted_images_left, undistorted_images_right

if __name__ == "__main__":
    CalibrateStereoCamera()
