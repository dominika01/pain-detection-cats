import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.models import Sequential, Conv2D, MaxPooling2D, Dense, Flatten
import glob
import os

# set up paths to folders
def setup_paths ():
    folder_dir = "data-keypoints"
    folders = ['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03', 
            'CAT_04', 'CAT_05', 'CAT_06']
    return folder_dir, folders

# displays an image with its keypoints
def display (image):
    keypoints = load_keypoints(image)
    img = mpimg.imread(image)
    imgplot = plt.imshow(img)
    plt.plot(*zip(*keypoints), marker='o', color='r', ls='')
    plt.show()

# displays an image with its actual and predicted keypoints
def display_prediction (image, keypoints, predictions):
    img = mpimg.imread(image)
    imgplot = plt.imshow(img)
    plt.plot(*zip(*keypoints), marker='o', color='g', ls='')
    plt.plot(*zip(*predictions), marker='o', color='r', ls='')
    plt.show()
    
# create an array of images from .jpg files
def load_images ():
    folder_dir, folders = setup_paths()
    images = []
    for folder in folders:
        path = os.path.join(folder_dir, folder, '*.jpg')
        images.extend(sorted(glob.glob(path)))
    return images
    
# load keypoints from a .cat file
def load_keypoints (path):
    path += '.cat'
    
    # split the file into an array
    # array contains: number of keypoints, x1, y1, x2, y2, ...
    with open(path, 'r') as f:
        line = f.read().split()
    
    
    keypointsNumber = int(line[0])
    keypoints = []
    i = 1
    
    # fill an array with keypoints
    while i < 2 * keypointsNumber:
        keypoints.append([int(line[i]), int(line[i+1])])
        i += 2
        
    return keypoints

# create train and test sets
def train_test_split (images):
    split = 0.8 * 9000
    
    x_train = images[:split]
    y_train = []
    
    x_test = images [split:]
    y_test = []
    
    for x in x_train:
        keypoints = load_keypoints(x)
        y_train.append(keypoints)
    
    for x in x_test:
        keypoints = load_keypoints(x)
        y_test.append(keypoints)
    
    return x_train, y_train, x_test, y_test

# very basic CNN model for now
def init_model ():
    model = Sequential()
    
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    return model

# train the model
def train (x_train, y_train):
    model = init_model()
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae','acc'])
    model.fit(x_train, y_train, batch_size=256, epochs=2, validation_split=2.0)
    return model

# test the model and display the 1st prediction    
def test (x_test, y_test, model):
    predictions = model.predict(x_test)
    display_prediction(x_test[0], y_test[0], predictions[0])

def main():
    images = load_images()
    x_train, y_train, x_test, y_test = train_test_split (images)
    model = train(x_train, y_train)
    test(x_test, y_test, model)