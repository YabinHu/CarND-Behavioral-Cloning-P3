import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(lines, test_size=0.25)

from sklearn.utils import shuffle
import random

def generator(samples, batch_size=32, in_train=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                if random.randint(0, 100) % 3 == 0: # flip 1/3 samples
                    center_image = np.fliplr(center_image)
                    center_angle = -center_angle
                images.append(center_image)
                angles.append(center_angle)

                if in_train:
                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(name)
                    left_angle = center_angle + correction
                    if random.randint(0, 100) % 3 == 0:
                        left_image = np.fliplr(left_image)
                        left_angle = -left_angle
                    images.append(left_image)
                    angles.append(left_angle)

                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(name)
                    right_angle = center_angle - correction
                    if random.randint(0, 100) % 3 == 0:
                        right_image = np.fliplr(right_image)
                        right_angle = -right_angle
                    images.append(right_image)
                    angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size,
                            in_train=False)
validation_generator = generator(validation_samples, batch_size=batch_size,
                                 in_train=False)

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Convolution2D, Dense, Flatten
from keras.layers import BatchNormalization, Lambda, Dropout
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

row, col, ch = 160, 320, 3 # For tf backend
new_height, new_width = 64, 64

model = Sequential()
model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x / 127.5 - 0.5, input_shape=(row - 90, col, ch),
                 output_shape=(row - 90, col, ch)))
model.add(Lambda(lambda x: K.resize_images(x, new_height, new_width, 'channels_last'),
                 input_shape=(row - 90, col, ch),
                 output_shape=(new_height, new_width, ch)))

model.add(Convolution2D(3, (1, 1), padding='same', name='color_conv'))
model.add(ZeroPadding2D((2, 2)))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(ZeroPadding2D((2, 2)))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(1, activation='softmax'))

model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(filepath="model_epoch{epoch:02d}.h5", verbose=1,
                              monitor='val_loss', save_best_only=True,
                              mode='min')
history = LossHistory()
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples)//batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)//batch_size,
                    epochs=10, callbacks=[checkpointer, history])

model.save('model.h5')
