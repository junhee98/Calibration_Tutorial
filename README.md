# Camera Calibration with python OpenCV

<!-- TOC -->


This repo contains a implementation of Camera-Calibration.
***
## Requirements

- numpy
- opencv-contrib-python (for Apriltag)
- Matplotlib
- pytransform3d


## Running
To conduct calibration, Put the set of images you want to calibrate into a single folder.
In this repo, puts the set of images want to calibrate in `images/*.jpg`.

```bash
python calib.py --src images/ --overlap True --plot True
```

Here are the parameters availble for calibration:
```
--src          The folder where the uncalibrated images. (e.g. images/)
--overlap      Overlapping distorted & undistort images for checking undistortion was conducted.
--plot         Plot Camera Transformation & Plane Transformation.
```


Then, you can find overlap images in `diff/*.jpg` and `overlap/*.jpg`.
And, you cand find plot images `Camera Transformation.png` and `Plane Transformation.png`
