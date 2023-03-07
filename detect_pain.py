
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
import pandas as pd
import tensorflow as tf
from keras.metrics import RootMeanSquaredError
from sklearn import model_selection


### DATA PREPROCESSING
def load_data():
    # set up the path
    csv_path = 'labels_preprocessed.csv'
    # read data into a pandas dataframe
    df = pd.read_csv(csv_path)
    return df

def preprocess_data():
    # load data and replace NaN values with 0s
    df = load_data()
    df.fillna(0)
    
    # split the data into x and y sets
    x_data = df.iloc[:, :-1] # df without the last column
    y_data = df.iloc[:, [0, -1]] # df only with the first and last column
    
    return x_data, y_data

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

preprocess_data()