## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/features_vis.png
[image3]: ./output_images/search_window.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/combine_boxes.png
[image7]: ./output_images/combined_labels.png
[image8]: ./output_images/false_positive.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 1st throught 5th code cells of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` image which is contained in the 2nd cell.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=36`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and histogram with 32 bins:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, includeing orientations, pixels per cell and cells per block. For eatch combination, I created dataset and then train the classifier to see the validation accuray.

The best accuracy I got is 98.73% by using 36 orientations, 8x8 pixels per cell and 2x2 cells per block.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using color features, histogram features and hog features. The code is at the 8th code cell in the notebook.

First, I normalize the features using `StandardScaler` to force all the features are between 0 and 1.

Then I use `train_test_split` to divide the dataset into train dataset and validation dataset, which would be used in cross validation.

Then I trained the linear SVM classifer using train dataset.

At last, I evalutated the classifier using `SVC.score` method by feeding the validation data.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decide to search in the image by using 3 scale of window, 1.5, 2.0, 2.5. Considering that the template size is 64x64, the window size mapped to the image is 96x96, 128x128, 160x160.

Another thing is the search region. I limited the search region to the bottom half of the image, where the road is mainly presented. The pixels of y is between 400 and 656. 

*Note: The image below is a just a diagram that showing what kind of window would be used. In reality, the step between two window would be just 16 pixels.*

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project-output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

And the labels of the combined heatmap is as below:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest challenge for me was choosing the features. At first, I tried to use `HSV` color map and I got a more than 99% test accuracy. But the result on the test image has too many false positive prediction. As the image below shows, on the left of the image, there was no car, but the model output positive:

![alt text][image8]

So I take a look of the train accuracy, then I found that the train accuracy was 100%. So I think the problem is **overfitting**. I think the training dataset is not enough. But because of my limited hardware, I cannot use the Udacity dataset. I believe if I could utilize more data, the result would be definitely better.

With these limitation, I tried other features, and `YCbCr` proved to be better in the project video challenge, so even its test accuracy is a little lower than `HSV`, I decide to use `YCbCr` as my final feature.

Another thing worth to mention is that when cars are little far from our car, they would possibly missed by the model. After some investigation, I found that is because there was only 1 window cound fit the car if the car is far away. Then the only window will be mark as false positive. So it looks like a dilimma on false positive and true positive.