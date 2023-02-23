import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# set up paths to folders
def setup_paths ():
    folder_dir = "data-keypoints"
    
    # TEMPORARY - to run on less data
    #folders = ['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03',
    #           'CAT_04', 'CAT_05', 'CAT_06']
    folders = ['CAT_00']
    return folder_dir, folders

# load & normalise the images and their corresponding keypoints,
def load_data():
    
    folder_dir, folders = setup_paths()
    images = []
    keypoints = []
    
    for folder in folders:
        path = os.path.join(folder_dir, folder, '*.jpg')
        images_paths = sorted(glob.glob(path))
        
        for image_path in images_paths:
            
            # load the image
            img = cv2.imread(image_path)
            original_size = img.shape[:2]
            
            # load keypoints
            with open(image_path+'.cat', 'r') as f:
                keypoints_text = f.read().strip()
            keypoints_arr = np.array(keypoints_text.split(' ')[1:], dtype=np.float32)
            keypoints_arr = keypoints_arr.reshape(-1, 2)

            # normalise the image and keypoints
            x_scale = 256 / img.shape[1]
            y_scale = 256 / img.shape[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(img, (256, 256))
            keypoints_arr = keypoints_arr * [x_scale, y_scale]

            # normalise pixel values
            image = image / 255.0
            keypoints_arr = keypoints_arr / 255.0

            # add images and keypoints to lists
            images.append(image)
            keypoints.append(keypoints_arr)

    # convert the lists to numpy arrays
    images = np.array(images)
    keypoints = np.array(keypoints)

    return images, keypoints, original_size

# display an image and its corresponding keypoint
def display_image(images, keypoints, index, original_size):
    
    # get the image and keypoints for the specified index
    # and normalise pixel values
    image = images[index] * 255
    keypoints = keypoints[index] * 255
    
    # resize the image and its keypoints
    keypoints = keypoints * [original_size[0]/256, original_size[1]/256]
    image = cv2.resize(image, original_size, interpolation=cv2.INTER_LINEAR)

    # display
    plt.imshow(image)
    plt.plot(*zip(*keypoints), marker='o', color='r', ls='')
    plt.show()

# define the CNN architecture - might improve later
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(18)
    ])
    return model

# run the code
def main():
    
    # TEMPORARY - testing load_data() and display_image()
    images, keypoints, original_size = load_data()
    display_image(images, keypoints, 0, original_size)
    
    return
    
main()