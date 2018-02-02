import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle

pickle_data = pickle.load(open('wide_dist_pickle.p', 'rb'))
mtx = pickle_data["mtx"]
dist = pickle_data["dist"]

img = cv2.imread('camera_cal/calibration5.jpg')
imgSize = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, imgSize[:2],None,None)

dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('output_images/test_undist.jpg',dst)

# Save the camera calibration result for later use
distPkl = {}
distPkl["mtx"] = mtx
distPkl["dist"] = dist
pickle.dump( distPkl, open( "output_images/wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()
