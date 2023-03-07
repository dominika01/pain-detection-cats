
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
import pandas as pd
import tensorflow as tf
from keras.metrics import RootMeanSquaredError
from sklearn import model_selection

INPUT_SHAPE = 256

### DATA PREPROCESSING
# preprocess an individual image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray
    img = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE)) #resize
    img = img / 255.0 #normalize pixel values
    return img

# load and preprocess the data
def load_data():
    ## LABELS
    # set up the path
    csv_path = 'labels_preprocessed.csv'
    # read data into a pandas dataframe
    labels = pd.read_csv(csv_path)
    # replace NaN values with 0s
    labels.fillna(0)
    
    ## EARS
    # set up the path
    ears_path = 'data/ears'
    ears_dir = os.listdir(ears_path)

    # iterate over images
    for image in ears_dir:
        # load the image
        image_path = os.path.join(ears_path, image)
        img = cv2.imread(image_path)
        # preprocess the image
        x_ears = []
        x_ears.append(preprocess_image(img))
        x_ears = np.array(x_ears)
        x_ears = np.expand_dims(x_ears, axis=-1)
    
    ## EYES
    # set up the path
    eyes_path = 'data/eyes'
    eyes_dir = os.listdir(eyes_path)

    # iterate over images
    for image in eyes_dir:
        # load the image
        image_path = os.path.join(eyes_path, image)
        img = cv2.imread(image_path)
        # preprocess the image
        x_eyes = []
        x_eyes.append(preprocess_image(img))
        x_eyes = np.array(x_eyes)
        x_eyes = np.expand_dims(x_eyes, axis=-1)
        
    ## MUZZLE
    # set up the path
    muzzle_path = 'data/muzzle'
    muzzle_dir = os.listdir(muzzle_path)

    # iterate over images
    for image in muzzle_dir:
        # load the image
        image_path = os.path.join(muzzle_path, image)
        img = cv2.imread(image_path)
        # preprocess the image
        x_muzzle = []
        x_muzzle.append(preprocess_image(img))
        x_muzzle = np.array(x_muzzle)
        x_muzzle = np.expand_dims(x_muzzle, axis=-1)
        
    return labels, x_ears, x_eyes, x_muzzle


def split_data(x_data, y_data):
    # split the dataset into train and test sets
    x_train_val, x_test, y_train_val, y_test = model_selection.train_test_split(
        x_data, y_data, test_size=0.2, random_state=42)

    # split the train and validation sets
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=42)
    
    return x_train, y_train, x_val, y_val, x_test, y_test


### BUILING & EVALUATING THE MODELS
def create_model():
    pass

def train_model():
    pass

def evaluate_model():
    pass


### SCORING EACH CATEGORY
def eyes():
    pass

def ears():
    pass

def muzzle():
    pass

def head():
    pass


### MAIN
def main():
    pass
