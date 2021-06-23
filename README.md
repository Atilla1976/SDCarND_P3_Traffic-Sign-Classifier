Jupyter Notebook
writeup_template.md
vor ein paar Sekunden
Markdown
File
Edit
View
Language
s
102
| Fully connected       | Output: 120                                   |
103
| RELU                  | Activation function                           |
104
| Dropout               | Keep_prob = 0,55                              |
105
| Fully connected       | Output: 84                                    |
106
| RELU                  | Activation function                           |
107
| Dropout               | Keep_prob = 0,55                              |
108
| Fully connected       | Output: 43                                    |
109
|                       |                                               | 
110
​
111
​
112
#### 3. How I trained my model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
113
​
114
The first architecture I tried was the shown above but without Dropout. By optimizing the hyperparameter - Epoch, Batchsize and learning rate - it was not possible to achieve an validation accuracy above 0.93. To train the model I choosed the following hyperparameter values
115
​
116
* Epoch: from 15  to 70
117
* Batchsize: from 64  to 196
118
* learning rate: from 0.0009 to 0.0011
119
​
120
Considering the computing time, I found with the first architecture an optimum in area Epoch = 25, Batchsize = 128 and learning rate = 0.00095.
121
​
122
Because of the big difference between the training accuracy (close to 1) and validation time I tried to implement Dropout. 
123
​
124
​
125
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
126
​
127
To get the validation set accuracy at least 0.93 I added Dropout-layers.
128
​
129
My final model results were:
130
* training set accuracy of 0,996
131
* validation set accuracy of 0,955
132
* test set accuracy of 0,937
133
​
134
​
135
A high accuracy on the training set but low accuracy on the validation set in the first architecture implies overfitting. Therefore, in a second step, I adjusted the architecture by including dropout. Here I choose several architectures: For example dropout layer after each activation function. But in the end the best validation accuracy results out of using dropout after fully connected-/Relu layers with a keep_prob value of 0.55.
136
​
137
 
138
​
139
### Testing the Model on New Images
140
​
141
#### 1. I choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
142
​
143
Here are five German traffic signs that I found on the web:
144
​
145
![alt text][image5]      ![alt text][image6]         ![alt text][image7]        ![alt text][image8]      ![alt text][image9]
146
​
147
​
148
The first image with a Speed limit 50 km/h sign might be difficult to classify because it was taken from obliquely below.
149
​
150
The second image with a No passing sign might not be difficult to classify because it is taken straight form the front under optimal light conditions.
151
​
152
The third image with a Yield sign might not be difficult to classify because it is the only one with an upside down triangle shape.
153
​
154
The fourth image with a Traffic signals sign might be difficult to classify because it was taken from obliquely below.
155
​
156
The fifth image with a Roundabout mandatory sign might not be difficult to classify it is taken straight form the front under optimal light conditions and the shape of the contents of all other round signs looks completly different.
157
​
158
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
159
​
160
Here are the results of the prediction:
161
​
162
| Image                 |     Prediction                                | 
163
|:---------------------:|:---------------------------------------------:| 
164
| Roundabout mandatory  | Roundabout mandatory                          | 
165
| Traffic signals       | Traffic signals                               |
166
| No passing            | No passing                                    |
167
| Speed limit 50 km/h   | speed limit 70 km/h                           |
168
| Yield                 | Yield                                         |
169
​
170
​
171
The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.937.
172
​
173
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
174
​
175
The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.
