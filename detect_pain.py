
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

# load the labels into a dataframe
def load_labels():
    # read data into a pandas dataframe
    csv_path = 'labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
    
    # replace NaN values with 0s
    labels.fillna(0)
  
# load the images and labels of ears  
def load_ears():
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
    
    # load labels
    labels = load_labels()
    y_ears = labels.iloc [:, [0,1]]
    
    return x_ears, y_ears

# load the images and labels of eyes 
def load_eyes():
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
    
    # load labels
    labels = load_labels()
    y_eyes = labels.iloc [:, [0,2]]
    
    return x_eyes, y_eyes
    
# load the images and labels of muzzles 
def load_muzzle():
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
    
    # load labels
    labels = load_labels()
    y_muzzle = labels.iloc [:, [0,3]]
    
    return x_muzzle, y_muzzle

# load the images and labels of whiskers 
def load_whiskers():
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
    
    # load labels
    labels = load_labels()
    y_whiskers = labels.iloc [:, [0,4]]

# load the images and labels of heads     
def load_head():
    # set up the path
    head_path = 'data/head'
    head_dir = os.listdir(head_path)

    # iterate over images
    for image in head_dir:
        # load the image
        image_path = os.path.join(head_path, image)
        img = cv2.imread(image_path)
        # preprocess the image
        x_head = []
        x_head.append(preprocess_image(img))
        x_head = np.array(x_head)
        x_head = np.expand_dims(x_head, axis=-1)
    
    # load labels
    labels = load_labels()
    y_head = labels.iloc [:, [0,5]]
    
    return x_head, y_head

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
