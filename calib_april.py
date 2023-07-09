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
# parser.add_argument('--overlap',required=True, help='over-lapping visualization')
# parser.add_argument('--plot', required=True, help='plot camera & checkerboard')
args = parser.parse_args()

# Source image path (Distort image)
src = args.src
un_dist = 'dist_april/'
if not os.path.exists(un_dist):
    os.makedirs(un_dist)

# Create the aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
arucoParams = cv2.aruco.DetectorParameters()
arucoParams.markerBorderBits = 2
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = [] # 3d point in real world space.
imgpoints = [] # 2d points in image plane.

# loading image list
images = sorted(glob.glob(src+'*.jpg'),key=lambda s: int(re.search(r'\d+', s).group()))

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.namedWindow('image_gray',cv2.WINDOW_NORMAL)
cv2.moveWindow('image', 20, 20)
cv2.moveWindow('image_gray', 1300,20)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.resizeWindow('image_gray',1280,1280)
    cv2.imshow('image_gray',gray)
    cv2.waitKey(100)
    # Find the AprilTag markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
    print(corners[1].shape)
    print(corners[1])
    print("+++++++++++++")
    if len(corners) > 0:
        objp = np.zeros((len(corners),1,3), np.float32)
        objpoints.append(objp)

        imgpoints.append(corners[0])

        # Draw and display the corners
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        cv2.resizeWindow('image',1280,1280)
        cv2.imshow('image',img)
        cv2.waitKey(100)
    else:
        print("No marker detected in ",fname)
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
with open('camera_parameter_april.pkl','wb') as f:
    pickle.dump(camera_parameter,f)

# if args.overlap:
#     overlapping(src, un_dist)

# if args.plot:
#     camera_transformation(CHECKERBOARD) # this should be updated based on the AprilTag marker
#     plane_transformation(CHECKERBOARD)  # this should be updated based on the AprilTag marker