import cv2
import numpy as np
import os
import glob
import re
import pickle
import argparse
from overlap import overlapping
from visualization import camera_transformation, plane_transformation

parser = argparse.ArgumentParser(description='Camera Calibration')
parser.add_argument('--src', required=True, help='Source image path')
parser.add_argument('--overlap',required=True, help='over-lapping visualization')
parser.add_argument('--plot', required=True, help='plot camera & checkerboard')
args = parser.parse_args()

#Source image path (Distort image) -- If you use custum data, change this path!
src = args.src
#Dist image path (Undistort image) -- If you use custum data, change this path!
un_dist = 'dist/'
if not os.path.exists(un_dist):
    os.makedirs(un_dist)

CHECKERBOARD = (9,13)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...,(6,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[1],0:CHECKERBOARD[0]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space.
imgpoints = [] # 2d points in image plane.

# loading image list
images = sorted(glob.glob(src+'*.jpg'),key=lambda s: int(re.search(r'\d+', s).group()))

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.moveWindow('image', 20, 20) 
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[1],CHECKERBOARD[0]), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        #Draw and display the corners
        img = cv2.drawChessboardCorners(img, (CHECKERBOARD[1],CHECKERBOARD[0]), corners2, ret)
        cv2.imshow('image',img)
        cv2.resizeWindow('image',640,640)
        cv2.waitKey(100)
    # If not found, print fname
    else:
        print("Not found obj point! : ",fname)
cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#Result
print("Error(Reprojection Error) : ",ret)
print("=============================\n")

print("Intrinsic_parameter : \n",mtx)
print("=============================\n")

print("distortion_coefficient : ",dist)
print("=============================\n")

print("Rotation : \n",rvecs)
print("=============================\n")

print("Translation : \n",tvecs)
print("=============================\n")

# Undistort image with calculated Camera parameter
cv2.namedWindow('undistorted img',cv2.WINDOW_NORMAL)
cv2.moveWindow('undistorted img', 700, 20)
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, new_camera_matrix)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    fname = fname.replace(src,'')
    cv2.imshow('undistorted img',dst)
    cv2.resizeWindow('undistorted img',640,640)
    cv2.waitKey(100)
    cv2.imwrite(os.path.join(un_dist,fname), dst)
cv2.destroyAllWindows()

# Refine Intrinsic parameter
print("new_camera_matrix : ",new_camera_matrix)


#Save calculated camera parameter
camera_parameter = {}

camera_parameter['instrinsic'] = new_camera_matrix
camera_parameter['distortion'] = dist
camera_parameter['rotation'] = rvecs
camera_parameter['translation'] = tvecs

# Save camera parameter
with open('camera_parameter.pkl','wb') as f:
    pickle.dump(camera_parameter,f)

if args.overlap:
    overlapping(src, un_dist)

if args.plot:
    camera_transformation(CHECKERBOARD)
    plane_transformation(CHECKERBOARD)
