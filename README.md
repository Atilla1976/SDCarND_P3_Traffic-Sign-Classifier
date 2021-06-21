# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Visualization_dataset.png "Visualization"
[image2]: ./images/all_traffic_signs.png "All traffic signs"
[image3]: ./images/random_noise.jpg "Random Noise"
[image4]: ./images/02_speed_limit_50.jpg "Traffic Sign 1"
[image5]: ./images/09_no_passing.jpg "Traffic Sign 2"
[image6]: ./images/13_Yield.jpg "Traffic Sign 3"
[image7]: ./images/26_traffic_signals.jpg "Traffic Sign 4"
[image8]: ./images/40_roundabout_mandatory.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://https://github.com/DerStuttgarter/SDCarND_P3_Traffic-Sign-Classifier/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32 x 32 pickels with 3 colour channels.
* The number of unique classes/labels in the data set is 43.

#### 2.Visualization of the dataset.

Here is an exploratory visualization of the data set. The three bar charts in the first row showing the frequency  of each traffic sign in the three datasets - training data, validation data and test data set.
The three bar charts in the second row showing the distribution  of each traffic sign in the three datasets. Here you can see that only in the test dataset the order of the traffic sign images are shuffled.


![alt text][image1]

The following images shows all types of traffic signs in the dataset.


![alt_text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because none of the traffic signs is only different in colour from another. Traffic signs can be distinguished from one another by their shape or the shape of the characters shown.

Here is an example of a traffic sign image before and after grayscaling and normalizing.

![alt text][image2]

Here is an example of a traffic sign image before and after grayscaling and normalizing.

![alt text][image2]

As a second step, I normalized the image data because .Among the best practices for training a Neural Network is to normalize the data to obtain a mean close to zero. Normalizing the input data generally speeds up learning. That is the reason, why I normelized the dataset in the second step. First I normalized the image data with "(pixel - 128)/128)". But I got better results by normalizing with "pixel/255".



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description								| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 5x5     	| 1x1 stride, "VALID"-padding, Output: 28x28x6 	|
| RELU					| Activation function, outputs 28x28x6			|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, "VALID"-padding, Output: 10x10x16	|
| RELU					| Activation function, Output: 10x10x16			|
| Max pooling	      	| 2x2 stride, Output: 5x5x16 					|
| Flatten				| Output: 400									|
| Fully connected		| Output: 120									|
| RELU					| Activation function							|
| Dropout				| Keep_prob = 0,55								|
| Fully connected		| Output: 84									|
| RELU					| Activation function							|
| Dropout				| Keep_prob = 0,55								|
| Fully connected		| Output: 43									|
|						|												| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The first architecture I tried was the shown above but without Dropout. By optimizing the hyperparameter - Epoch, Batchsize and learning rate - it was not possible to achieve an validation accuracy above 0.93. There were also big differences to the training accuracy (close to 1). Because of the 

To train the model, I used in a

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0,996
* validation set accuracy of 0,955
* test set accuracy of 0,937


The first architecture I tried was the shown above but without Dropout. By optimizing the hyperparameter - Epoch, Batchsize and learning rate - it was not possible to achieve an validation accuracy above 0.93. There were also big differences to the training accuracy (close to 1).
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory	| Roundabout mandatory							| 
| Traffic signals		| Traffic signals								|
| No passing			| No passing									|
| speed limit 50 km/h	| speed limit 70 km/h			 				|
| Yield					| Yield											|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.937.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Roundabout mandatory sign (probability of 0.997), and the image does contain a Roundabout mandatory sign. The top five soft max probabilities were:

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| .9973					| Roundabout mandatory							| 
| .0011					| No passing for vehicles over 3.5 metric tons	|
| .0011					| Vehicles over 3.5 metric tons prohibited		|
| .0003					| End of all speed and passing limits			|
| .0002					| Children crossing								|



For the second image, the model is relatively sure that this is a Traffic signals sign (probability of 0.998), and the image does contain a Traffic signals sign. The top five soft max probabilities were:

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| .9983					| Traffic signals								| 
| .0017					| General caution								|
| <.0000				| Road narrows on the right						|
| <.0000				| Pedestrians									|
| <.0000				| Bicycles crossing								|



For the third image, the model is absolutely sure that this is a No passing sign (probability of nearly 1.0), and the image does contain a No passing sign. The top five soft max probabilities were:

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| 1.0					| No passing									| 
| <.0000				| No passing for vehicles over 3.5 metric tons	|
| <.0000				| Vehicles over 3.5 metric tons prohibited		|
| <.0000				| Dangerous curve to the right					|
| <.0000				| End of no passing								|



For the fourth image, the model is a little sure that this is a Speed limit 70km/h sign (probability of 0.68), but the image does contain a Speed limit 50km/h sign. The top five soft max probabilities were:

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| .6798					| Speed limit (70km/h)							| 
| .3127					| Traffic signals								|
| .0065					| Speed limit (120km/h)							|
| .0004					| Roundabout mandatory							|
| .0003					| General caution								|



For the fifth image, the model is absolutely sure that this is a Yield sign (probability of nearly 1.0), but the image does contain a Yield sign. The top five soft max probabilities were:

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| 1.0					| Yield											| 
| <.0000				| Speed limit (20km/h)							|
| <.0000				| Speed limit (30km/h)							|
| <.0000				| Speed limit (50km/h)							|
| <.0000				| Speed limit (60km/h)							|
