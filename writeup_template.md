##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./write_up_images/vehicle_non_vehicle.png
[image2]: ./write_up_images/hog_image.png
[image3]: ./write_up_images/windows_inefficient.png
[image4]: ./write_up_images/windows_efficient.png
[image5]: ./write_up_images/heatmap.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

Here is an overview of what I'm covering in this project:

* 1) Feature extraction
* 2) Train a vehicle/non-vehicle classifier
* 3) Apply classifer to project video to draw boxes around vehicles on the road

After going over the project lecture videos, I decided to use a combination of Histograms of color, spatial binning of color and Histogram of Oriented Gradients (HOG) as my feature extraction techniques. For my classifier, I decided to use a Linear SVM to as a starting point as the lecturer, Arpan, recommended using this machine learning tool. Finally for the last step, I'm using HOG sub-sampling window search along with histograms of color and spatial binning to predict vehicles on a given image, or video, of the road.

The entire project code base is in the a file name `model.ipynb`.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook, using a method called `get_hog_features` in the `model.ipynb` file. The actual feature extraction is done in a method named `extract_features`, where it extract features using **histograms of color** and **spatial binning**. Within `extract_features`, I read through the vehicle images in `KITTI_extracted` folder and the non-vehicle images in `Extras` folder. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![vehicle/non-vehicle][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orient`, `pix_per_cell`, `hog_channel` and `cell_per_block`). 

![HOG image][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Initially I started with the following combination of parameters:

```python
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL", each looking at different color channel, or 'ALL' for all channels
```

As I went through the project lecture videos, I experimented with many different combinations of these parameters and these seemed to produce the most robust result. However, as I discovered later, despite having high training/validation accuracy, my model performed poorly on the `test_video.mp4` video and the actual project video, `project_video.mp4`. One immediate improvement was to change the `color_space` parameter from `RGB` to `HSV` as recommended in the lecture because vehicles seem to have distinct saturation feautres, so here is what I settled on:

```python
color_space = 'HSV'
orient = 9  
pix_per_cell = 8 
cell_per_block = 2 
hog_channel = "ALL"
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a combination of HOG, histograms of colors and spatial binning as outlined above. The code for training the classifier is in the 5th code cell in the `model.ipynb` notebook. I extracted vehicle and non-vehicle features and combined the 3 features into a single vector before performing feature scaling to make sure that we don't have a single feature that dominates the rest. My model achieved an accuracy rate of 99.73% on the test dataset.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I initially implemented an inefficient sliding window search technique, namely, it would extract HOG features in every search window. This code is in the 6th code cell in the `model.ipynb` notebook, where I defined the `slide_window`, `single_img_features`, and `search_windows` methods. 

The implementation of these methods are in the 8th cell of the `model.ipynb` notebook, under **Draw hot windows on test images**. I initially used the `slide_window` method to get a list of windows with various sizes (but only in areas of the image where you'd expect to see vehicles at all), ranging from `56px` to `104px`, then used the `search_windows` method to loop through these windows of various sizes and determine which of these windows contain a vehicle, via the `single_img_features` method.

Here is an output of the windows that my model determines to contain vehicles:

![raw hot windows inefficient][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The inefficient algorithm described above was indeed very slow. It took an average of **85 seconds** to extract features from a test image and then apply the sliding windows technique to it. This is way to slow and if I apply this to a one minute long video, it would have taken over 30 hours to complete! 

In order to improve it, I decided to use the **HOG sub-sampling** technique. This technique is a lot more efficient because it only extracts the HOG features once, as opposed to doing that in every window. This drastically reduced the amount of time taken to run the algorithm, so it only took about 5 seconds to run sliding windows technique on an image, and would take about 3 hours to run it over the project video. Still not perfect but a lot faster than than inefficient algorithm.

Here is an example of the output image, with 144 windows:

![raw hot windows efficient][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

After recording the positions of the positive detections in each frame of the video, I created a heatmap and then thresholded (using a threshold of 12) that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Assuming each blob is a vehicle, I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a test image, the result of `scipy.ndimage.measurements.label()` and the bounding boxes on the final positions of the detected vehicles:

![heatmap][image5]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the sliding window search section, I fixated the starting y position where I begin to search for vehicles so that I can filter out the areas, such as the sky, that you do not expect to see vehicles. However, on the image/video captuerd by a different camera, this will likely not work since the starting y position may be different. It would be nice to dynamically change the starting y position based on the characteristics of the particular image/video.

Another problem I encountered was that even the 'efficient' sliding window algorithm using HOG sub-sampling was slow (more than 5 seconds to draw windows for a single image frame). This would likely fail in practice. I would need to find out further ways of improving my algorithm to make it faster, one possible approach is that I used windows of various sizes across the entire searchable area in the image/video to detect vehicles, but the various window sizes should be based on where you are in the image, i.e. if the search area is closer to the bottom of the screen then the window should be larger, this should further improve my algorithm.
