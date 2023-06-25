import cv2 
import os
import glob
import numpy as np


def overlapping(src, un_dist):
    images = glob.glob(src+'*jpg')
    cv2.namedWindow('distort_img',cv2.WINDOW_NORMAL)
    cv2.moveWindow('distort_img', 20, 20)
    cv2.namedWindow('undistort_img',cv2.WINDOW_NORMAL)
    cv2.moveWindow('undistort_img', 700, 20)
    cv2.namedWindow('overlap_img',cv2.WINDOW_NORMAL)
    cv2.moveWindow('overlap_img', 20, 700)
    cv2.namedWindow('diff_img',cv2.WINDOW_NORMAL)
    cv2.moveWindow('diff_img', 700, 700)

    diff = 'diff/'
    if not os.path.exists(diff):
        os.makedirs(diff)
    overlap = 'overlap/'
    if not os.path.exists(overlap):
        os.makedirs(overlap)

    for fname in images:
        dist_img = cv2.imread(fname)
        cv2.imshow('distort_img',dist_img)
        cv2.resizeWindow('distort_img',640,640)

        fname = fname.replace(src,'')
        undist_img = cv2.imread(os.path.join(un_dist,fname))
        cv2.imshow('undistort_img',undist_img)
        cv2.resizeWindow('undistort_img',640,640)

        dist_img = cv2.resize(dist_img, (undist_img.shape[1],undist_img.shape[0]))

        # Compute the absolute difference between the images
        diff_img = cv2.absdiff(dist_img,undist_img)
        diff_img = cv2.convertScaleAbs(diff_img, alpha=3)
        cv2.imshow('diff_img',diff_img)
        cv2.resizeWindow('diff_img',640,640)
        cv2.imwrite(os.path.join(diff,fname),diff_img)
        #cv2.waitKey(0)

        # Overlap dist & undist images
        overlap_img = cv2.addWeighted(dist_img, 0.2,undist_img, 0.8, 0)
        cv2.imshow('overlap_img',overlap_img)
        cv2.resizeWindow('overlap_img',640,640)
        cv2.imwrite(os.path.join(overlap,fname),overlap_img)
        cv2.waitKey(100)


    cv2.destroyAllWindows()