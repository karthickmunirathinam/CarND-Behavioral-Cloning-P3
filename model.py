import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import logging
from keras.utils import plot_model


logging.basicConfig(level=logging.INFO)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            # For use of multiple camera images, can be possible if we slightly correct
            # steering angles for left and right camera images by say 0.2 factor
            # e.g. corrected_left_steering_angle = left_steering_angle + 0.2
            #      corrected_right_steering_angle = right_steering_angle - 0.2
            # keeping center steering angle as it is.
            # correct_factor = [center, left, right]
            correct_factor = [0.0, 0.2, -0.2]
            for batch_sample in batch_samples:
                for i in range(3):  # 0. Center, 1. Left, 2. Right
                    name = batch_sample[i]

                    image = cv2.imread(name)
                    if image is None:
                        logging.error("Failed to open %s" % (name))
                        continue
                    # OpenCV opens image in BGR mode whereas drive.py uses it as RGB
                    # Considering conversion of training image to RGB makes some impact
                    # on test performance/accuracy.
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[3]) + correct_factor[i]
                    images.append(image)
                    angles.append(angle)
                    # Augment Dataset by flip image and negative angle for increasing
                    # dataset
                    images.append(cv2.flip(image, 1))
                    angles.append(angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()

    lines = []
    with open('driving_log.csv') as fp:
        reader = csv.reader(fp)
        for line in reader:
            lines.append(line)

    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    logging.info("number of training samples: %s" % (len(train_samples)))
    logging.info("number of validation samples: %s" % (len(validation_samples)))

    train_generator = generator(train_samples, 32)
    validation_generator = generator(validation_samples, 32)

    # --- NVIDIA Model ---
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Convolution2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                        validation_data=validation_generator, validation_steps=len(validation_samples), epochs=2, verbose=1)

    model.save('model.h5')
