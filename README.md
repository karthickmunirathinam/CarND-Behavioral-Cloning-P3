
# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_net.png "Model Architecture"
[image2]: ./examples/center_2018_10_22_21_05_13_424.jpg "Captured Image"
[image3]: ./examples/left_2018_10_22_21_05_13_424.jpg "Captured Image"
[image4]: ./examples/right_2018_10_22_21_05_13_424.jpg "Captured Image"
[image5]: ./examples/left_2018_10_22_21_33_08_411.jpg "Normal Image"
[image6]: ./examples/left_2018_10_22_21_33_08_411_fliped.jpg "Flipped Image"

### Required files and Quality of code

My project includes thew following files according to the acceptance criteria. 

* model.py  -Will create and train the model 
* drive.py - Will drive the car in autonomous mode with few parameters modified
* model.h5 -Will contain a trained convolution neural network
* writeup_report.md -Will summarize the results
* video.py - Will help to record the simulation video
* video.mp4 - Video of the car driving at autonomous mode.

From the provided files and simulator from Udacity. The model is trained from the training set obtained by driving the car in the simulator mode. The car can autonomously drive along the circuit without crossing the road boundaries.
We can this the below script to observe the result.
```sh
python drive.py model.h5
```

## Model Architecture and Training Strategy

My primary influence of model architecture is from this paper below.
('https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf')

My network architecture is adopted from the Nvidia Net. This model is choosen because it converges well for regression problem like this here predicting steering angle.

 ![alt text][image1]

### Criteria:
1. My model is CNN with 3x3 and 5x5 filter sizes and depths between 24 and 100.
2. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.
3. The model contains dropout layers to reduce overfitting.
4. The data set is split into training set and Validation set to avoid overfitting.
5. Adams optimizer was used so the learning rate was not tuned manually.
6. Appropriate training datas are choosen to ensure the car drives in the center of the road for the complete circuit.

## Architecture and Training Documentation

1. The first step was to divide my images and steering angle data into a training and validation set.
2. Effectively predict the steering angle by seeing center, left and right camera images.
3. I added Dropout layers after observing low error in training set and high error in validation set, which is a clear sign of overfitting.
4. The big challenge I faced was in turnings especially in sharp turns, where the car often moves out of the road boundary. I have solved the problem by taking several set of data at sharp turns (The car moving from road boundary to the middle). Doing so I acheived smoother steering angle at sharp turns while remaining at road center.
5. The Final architecture is shown below

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
| **Number of Epochs:** 2       |                     |         |

6. For training, I have first acquired 2 laps on track driving on center lane. Then I have acquired 2 laps but in the opposite direction for better generalization. 
7. For the scenarios where the vehicle has to recover from the left side and right side edges to the center of the road, I have recorded several vehicle motion especially in the turning (both right and left turns evenly). This shown in the figure below along with 3 camera images (left, middle and right).

![alt text][image2]
![alt text][image3]
![alt text][image4]

8. I have corrected steering angle for left and right camera images by 0.2 and -0.2, respectively for allowing model to correct steering angle more promptly.
9. I have also flipped images and angles to append due to the fact that the circuit has more left turns than right turns. An example is shown below.

![alt text][image5]
![alt text][image6]

10. The images were cropped by removing top 70 rows and bottom 25 rows so the model learns better.
11. I finally randomly shuffled the data set and put 20% of the data into a validation set.


## Simulation

The car suceesfully drove in the autonomous mode without moving out of the driving surface. The maximum speed this was achievable was upto 20 mph. Beyond this speed the car vobbles and moves out of the road boundary.


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.



