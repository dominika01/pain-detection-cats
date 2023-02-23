import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn
from sklearn import model_selection
import glob
import os

# set up paths to folders
def setup_paths():
    folder_dir = "data-keypoints"
    
    # TEMPORARY - to run on less data
    #folders = ['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03',
    #           'CAT_04', 'CAT_05', 'CAT_06']
    folders = ['CAT_00']
    return folder_dir, folders

# Load the images and their corresponding key points
def load_data():
    folder_dir, folders = setup_paths()
    images = []
    keypoints = []
    for folder in folders:
        path = os.path.join(folder_dir, folder, '*.jpg')
        images_paths = sorted(glob.glob(path))
        
        for image_path in images_paths:
            # Load the image
            img = cv2.imread(image_path)
            original_size = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # load keypoints
            with open(image_path+'.cat', 'r') as f:
                keypoints_text = f.read().strip()

            # Parse the keypoints text
            keypoints_arr = np.array(keypoints_text.split(' ')[1:], dtype=np.float32)
            keypoints_arr = keypoints_arr.reshape(-1, 2)

            # Basic pre processing:
            # Resize the image and keypoints
            x_scale = 256 / img.shape[1]
            y_scale = 256 / img.shape[0]
            image = cv2.resize(img, (256, 256))
            keypoints_arr = keypoints_arr * [x_scale, y_scale]

            # Normalize the image and keypoints
            image = image / 255.0
            keypoints_arr = keypoints_arr / 255.0

            # Add the image and keypoints to the lists
            images.append(image)
            keypoints.append(keypoints_arr)
            
    return images, keypoints, original_size

# pre-process the data
def preprocess_data():
    images, keypoints, size = load_data()
    
    # Convert the lists to numpy arrays
    images = np.array(images)
    keypoints = np.array(keypoints)
    
    # reshape the arrays
    images = np.expand_dims(images, axis=-1)
    keypoints = keypoints.reshape(len(keypoints),18)
    
    # split the dataset into train and test sets
    train_val_images, test_images, train_val_keypoints, test_keypoints = model_selection.train_test_split(
        images, keypoints, test_size=0.2, random_state=42)

    # split the train and validation sets
    train_images, val_images, train_keypoints, val_keypoints = model_selection.train_test_split(
        train_val_images, train_val_keypoints, test_size=0.2, random_state=42)
    
    # set up data generators for the sets
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(
        train_images, train_keypoints, batch_size=32)

    validation_generator = val_datagen.flow(
        val_images, val_keypoints, batch_size=32)
    
    test_generator = test_datagen.flow(
        test_images, test_keypoints, batch_size=32)
    
    return train_generator, test_generator, validation_generator

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

# train the model
def train_model(train, val):
    model = create_model()
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mae'])
    history = model.fit(train, epochs=10, validation_data=val)
    model.save('models/model_1') #save the model to a file
        
    return history, model

# evaluate the model
def evaluate_model(model, test):
    loss, mae = model.evaluate(test)
    return loss, mae

# run the code
def main():
    train, test, val = preprocess_data()
    history, model = train_model(train, val)
    print(history)
    loss, mae = evaluate_model(model, test)
    print(loss)
    print(mae)
    
main()