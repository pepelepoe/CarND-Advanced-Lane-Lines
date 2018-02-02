import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import glob

# Map the coordinates of the corners in 2D image (imgpoints) to the 3D coordinates
# of the real undistorted chessboard corners (objpoints) 
nx = 9
ny = 6

# Read in the saved objpoints and imgpoints
objp = np.zeros((ny*nx, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objPoints = [] # 3D points in real world space
imgPoints = [] # 2D points in image plane

# List of calibration image_list
imageList = glob.glob('camera_cal/calibration*.jpg')

# Iterate through list and search for chessboard corners
for idx, fname in enumerate(imageList):
    img = cv2.imread(fname)
    # print(fname)
    # plt.imshow(img)
    # plt.show()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    print(ret)

    # If found, add object points and image points
    if ret == True:
        objPoints.append(objp)
        imgPoints.append(corners)

        # Draw and display the findChessboardCorners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        saveDir = 'output_images/'
        writeName = saveDir + 'corners_found'+str(idx)+'.jpg'
        cv2.imwrite(writeName, img)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()
