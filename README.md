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
[image3]: ./images/gry_norm_traffic_signal_sign.png "greyscaled and normalized traffic signal sign"
[image4]: ./images/shuffled_distribution.png "Shuffled distribution of training and validation data"
[image5]: ./images/LeNet.png "LeNet architecture"
[image6]: ./images/Acc_E25_B124_R00095_DO_ConvFullc_Tr1_0_Ev0_55.png "Training Validation Test Accuracy"
[image7]: ./images/02_speed_limit_50.jpg "Traffic Sign 1"
[image8]: ./images/09_no_passing.jpg "Traffic Sign 2"
[image9]: ./images/13_Yield.jpg "Traffic Sign 3"
[image10]: ./images/26_traffic_signals.jpg "Traffic Sign 4"
[image11]: ./images/40_roundabout_mandatory.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/DerStuttgarter/SDCarND_P3_Traffic-Sign-Classifier/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Providing a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32 x 32 pickels with 3 colour channels.
* The number of unique classes/labels in the data set is 43.

#### 2.Visualization of the dataset.

Here is a visualization of the data set. The three bar charts in the first row showing the frequency  of each traffic sign in the three datasets - training data, validation data and test data set.
The three bar charts in the second row showing the distribution of each traffic sign in the three datasets. Here you can see that only in the test dataset the order of the traffic sign images are shuffled.


![alt text][image1]

The following images shows all 43 classes of traffic signs in the training dataset.


![alt_text][image2]


### Design and Test a Model Architecture

#### 1. Preprocessing the image data.

#### What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because none of the traffic signs is only different in colour from another. Traffic signs can be distinguished from one another by their shape or the shape of the content shown.

As a second step, I normalized the image data. Among the best practices for training a Neural Network is to normalize the data to obtain a mean close to zero. Normalizing the input data generally speeds up learning. 
First I normalized the image data with "(pixel - 128)/128". But I got better results by normalizing with "pixel/255".

Here is an example of an image with a traffic signal sign after grayscaling and normalizing.

![alt text][image3]

In a third and last step, I shuffeled the order of the data sets because they were sorted as you can see in the training- and validations diagrams above.

![alt text][image4]

#### 2. Final model architecture

#### looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

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


#### 3. How I trained my model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The first architecture I tried was the LeNet architecture (with out the dropout-layers) because it is small in term of memory footprint and is made to recognize objects in images.

![alt text][image5]


By optimizing the hyperparameter - Epoch, Batchsize and learning rate - it was not possible to achieve an validation accuracy above 0.93. To train the model I choosed the following hyperparameter values

* Epoch: from 15  to 70
* Batchsize: from 64  to 196
* learning rate: from 0.0009 to 0.0011

Considering the computing time, I found with the first architecture an optimum in area Epoch = 25, Batchsize = 128 and learning rate = 0.00095.



The high accuracy on the training set (very close to 1) but low accuracy on the validation set (around 0.9) in the first architecture implies overfitting. Therefore, in a second step, I adjusted the architecture by including dropout-layers. Here I choose several architectures: For example dropout layer after each activation function. But in the end the best validation accuracy results out of using dropout only after fully connected-/Relu layers with a keep_prob value of 0.55.

My final model results were:
* training set accuracy of 0,996
* validation set accuracy of 0,955
* test set accuracy of 0,937

![alt text][image6]

### Testing the Model on New Images

#### 1. I choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]		 ![alt text][image8]		 ![alt text][image9]		![alt text][image10]		 ![alt text][image11]


The first image with a Speed limit 50 km/h sign might be difficult to classify because it was taken from obliquely below.

The second image with a No passing sign might not be difficult to classify because it is taken straight form the front under optimal light conditions.

The third image with a Yield sign might not be difficult to classify because it is the only one with an upside down triangle shape.

The fourth image with a Traffic signals sign might be difficult to classify because it was taken from obliquely below.

The fifth image with a Roundabout mandatory sign might not be difficult to classify it is taken straight form the front under optimal light conditions and the shape of the contents of all other round signs looks completly different.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory	| Roundabout mandatory							| 
| Traffic signals		| Traffic signals								|
| No passing			| No passing									|
| Speed limit 50 km/h	| speed limit 70 km/h			 				|
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
