import cv2
import numpy as np

CHECKERBOARD = (9,13)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...,(6,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[1],0:CHECKERBOARD[0]].T.reshape(-1,2)

print(objp.shape)

objp_2 = np.zeros((24,1,3), np.float32)


print(objp_2.shape)