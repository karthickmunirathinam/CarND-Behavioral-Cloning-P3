# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_net.png "Model Architecture"
[image2]: ./examples/left_2018_10_22_21_05_13_424.jpg "Captured Image"
[image3]: ./examples/center_2018_10_22_21_05_13_424.jpg "Captured Image"
[image4]: ./examples/right_2018_10_22_21_05_13_424.jpg "Captured Image"
[image5]: ./examples/left_2018_10_22_21_33_08_411.jpg "Normal Image"
[image6]: ./examples/left_2018_10_22_21_33_08_411_fliped.jpg "Flipped Image"
[image7]: ./examples/center_2018_10_22_21_05_34_522.jpg "Center of lane"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py  -Will create and train the model 
* drive.py - Will drive the car in autonomous mode with few parameters modified
* model.h5 -Will contain a trained convolution neural network
* writeup_report.md -Will summarize the results
* video.py - Will help to record the simulation video
* video.mp4 - Video of the car driving at autonomous mode.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 100 (model.py lines 80-98). My network architecture is adopted from the Nvidia Net. This model is choosen because it converges well for regression problem like this here predicting steering angle.
Its was made open source- [Link]('https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf').

 ![alt text][image1]

The model includes RELU layers to introduce nonlinearity (model.py line 80-98), and the data is normalized in the model using a Keras lambda layer (model.py line 81).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 80-98).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 84). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to predict the steering angle based on center, left and right camera images.

My first step was to use a convolution neural network model similar to the the Nvidia Net. I thought this model might be appropriate because it converges well for regression problem which in our case is predicting steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it does not overfit. For this I have added dropout layers after each convolution Layer.
Then I added appropriate training datas which are choosen to ensure the car drives in the center of the road for the complete circuit. For generalization I accumulated data in the reverve direction of the circuit.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I took several set of data at sharp turns (The car moving from road boundary to the middle). Doing so I acheived smoother steering angle at sharp turns while remaining at road center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 80-98) consisted of a convolution neural network with the following layers and layer sizes.

| Layer (type)                  | Output Shape        | Param # |
| ----------------------------- | ------------------- | ------- |
| lambda_1 (Lambda)             | (None, 160, 320, 3) | 0       |
| cropping2d_1 (Cropping2D)     | (None, 65, 320, 3)  | 0       |
| conv2d_1 (Conv2D)             | (None, 31, 158, 24) | 1824    |
| dropout_1 (Dropout)           | (None, 31, 158, 24) | 0       |
| conv2d_2 (Conv2D)             | (None, 14, 77, 36)  | 21636   |
| dropout_2 (Dropout)           | (None, 14, 77, 36)  | 0       |
| conv2d_3 (Conv2D)             | (None, 5, 37, 48)   | 43248   |
| dropout_3 (Dropout)           | (None, 5, 37, 48)   | 0       |
| conv2d_4 (Conv2D)             | (None, 3, 35, 64)   | 27712   |
| dropout_4 (Dropout)           | (None, 3, 35, 64)   | 0       |
| conv2d_5 (Conv2D)             | (None, 1, 33, 64)   | 36928   |
| dropout_5 (Dropout)           | (None, 1, 33, 64)   | 0       |
| flatten_1 (Flatten)           | (None, 2112)        | 0       |
| dense_1 (Dense)               | (None, 100)         | 211300  |
| dense_2 (Dense)               | (None, 50)          | 5050    |
| dense_3 (Dense)               | (None, 10)          | 510     |
| dense_4 (Dense)               | (None, 1)           | 11      |
|                               |                     |         |
| **Total params:** 348,219     |                     |         |
| **Trainable params:** 348,219 |                     |         |
| **Non-trainable params:** 0   |                     |         |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image7]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the edges of track. These images show what a recovery looks like starting from right where the car is placed and moved to the center (left, center and right).

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generate datas which could avoid the model to be biased only to the left turn as the circuit has more ledft turns. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

Then I have acquired 2 laps but in the opposite direction for better generalization.

After the collection process, I had 4841 number of data points. I then preprocessed this data by cropping the image by removing top 70 rows (background like trees,etc) and bottom 25 rows(car) so the model can learn better.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by me. I used an adam optimizer so that manually training the learning rate wasn't necessary.
