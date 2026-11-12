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
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
import cv2
from numpy.linalg import inv, pinv

def CalibrateStereoCamera():
    """
    Calibrates a stereo camera setup using images of a chessboard pattern.

    Parameters:
    - left_images: List of file paths to the left camera images.
    - right_images: List of file paths to the right camera images.
    """
    
    img_L = glob.glob('0_calib/left/*.png')
    img_R = glob.glob('0_calib/right/*.png')

    nb_vertical = 5
    nb_horizontal = 7

    # Left camera calibration
    objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
    objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_L = [] # 2d points in image plane.

    for fname in img_L:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h,  w = img.shape[:2]

        ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal))

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints_L.append(corners)

    ret_L, mtx_L, dist_L, rvecs_L, tvecs_L = cv2.calibrateCamera(objpoints, imgpoints_L, gray.shape[::-1], None, None)


    # Right camera calibration
    objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
    objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_R = [] # 2d points in image plane.

    for fname in img_R:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h,  w = img.shape[:2]

        ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal))

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints_R.append(corners)

    ret_R, mtx_R, dist_R, rvecs_R, tvecs_R = cv2.calibrateCamera(objpoints, imgpoints_R, gray.shape[::-1], None, None)

    # Stereo calibration
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_L, imgpoints_R, mtx_L, dist_L, mtx_R, dist_R, (w,h), flags=cv2.CALIB_FIX_INTRINSIC)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_L, dist_L, mtx_R, dist_R, (w,h), R, T)

    # Compute the undistortion and rectification transformation maps
    mapLx, mapLy = cv2.initUndistortRectifyMap(mtx_L, dist_L, R1, P1, (w, h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(mtx_R, dist_R, R2, P2, (w, h), cv2.CV_32FC1)

    # Save calibration values and maps to compressed .npz files
    np.savez_compressed("calibration_values.npz",
    ret_R=ret_R,
    mtx_R=mtx_R,
    dist_R=dist_R,
    rvecs_R=np.asarray(rvecs_R),
    tvecs_R=np.asarray(tvecs_R),
    ret_L=ret_L,
    mtx_L=mtx_L,
    dist_L=dist_L,
    rvecs_L=np.asarray(rvecs_L),
    tvecs_L=np.asarray(tvecs_L),
    ret=ret,
    R=R,
    T=T,
    E=E,
    F=F,
    R1=R1,
    R2=R2,
    P1=P1,
    P2=P2,
    Q=Q,
    roi1=np.asarray(roi1),
    roi2=np.asarray(roi2))

    np.savez_compressed("calibration_maps.npz",
    mapLx=mapLx,
    mapLy=mapLy,
    mapRx=mapRx,
    mapRy=mapRy
    )

    return True

CalibrateStereoCamera()
