## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_undistorted.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_combination.png "Binary Example"
[image4]: ./output_images/perspectivetransform.png "Warp Example"
[image5]: ./output_images/lane-pixels-fit.jpg "Fit Visual"
[image6]: ./output_images/find_lines_output.jpg "Output"
[video1]: ./output_images/output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines # through # of the file called `camera_cal.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in the file called `camera_undistort.py`.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of gradient threshold in x combined color transform using S channel to generate a binary image (thresholding steps at lines 38 through 58 in `advLineFinding.py`). Here's an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_perspective()` and `perspective_transform()`, which appear in lines 145 through 153 in the file `advLineFinding.ipynb`. The first function takes as inputs an image (`img`) and `M` returned by the second function. The second function takes as input (`src`) and destination (`dst`). I chose the hardcoded source and destination points in the `get_birds_eye()` (lines 156 through 166) function in the following manner:

```python
  top_left = [570,470]
  top_right = [720,470]
  bottom_right = [1130,720]
  bottom_left = [200,720]
  pts = np.array([bottom_left,bottom_right,top_right,top_left])

  top_left_dst = [320,0]
  top_right_dst = [980,0]
  bottom_right_dst = [980,720]
  bottom_left_dst = [320,720]
  dst_pts = np.array([bottom_left_dst, bottom_right_dst, top_right_dst, top_left_dst])

  src = np.float32(pts.tolist())
  dst = np.float32(dst_pts.tolist())
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 320, 720      |
| 1130, 720     | 980, 720      |
| 720, 470      | 980, 0        |
| 570, 470      | 320, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

On the `sliding_window_search()` function (lines 12 through 107 on `advLineFinding.ipynb` ) is where I perform lane-line pixel search by creating a histogram a doing sliding window search with 9 windows. I then iterate through each window and extract pixel positions and finish off by fitting a second order polynomial to left and right lines.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 109 through 143 in `sliding_window_search()` function on `advLineFinding.ipynb` code. I calculated the x position of y at the the height of the image for each lane. Afterwards I calculated the lane midpoint and converted to meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 246 through 273 in `find_lines()` function on `advLineFinding.ipynb` script.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

<!-- Here's a [link to my video result](./project_video.mp4) -->

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think I need to improve the way I calculated the radius of curvature and the position of the vehicle with respect to center.

#TODO:
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

For my approach, I created a function called find_lines which takes an input image and runs the project pipeline. I start the pipeline with image thresholding where I used a combination of Sobel x gradient thresholding with color transforms. For the latter I implemented L of CIELUV color and B of CIELAB color space. After thresholding, binary image is warped by `get_birds_eye()` function. In this function I harcoded appropriate source and destination points which I then used to obtain a perspective transform matrix with `perspective_transform()` function. I then input this matrix into `warp_perspective()` which returns a warped binary image or bird's eye view. This bird's eye view will be helpful in order to fit polynomial to lane lines 
