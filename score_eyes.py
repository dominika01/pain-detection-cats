
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn import model_selection

# global variables
INPUT_SHAPE_X = 128
INPUT_SHAPE_Y = 64
ITERATION = '1'
EPOCHS = 8
BATCH_SIZE = 32

### ERROR: there are too many pictures somehow???

### DATA PREPROCESSING - works
# preprocess an individual image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray
    img = cv2.resize(img, (INPUT_SHAPE_X, INPUT_SHAPE_Y)) #resize
    img = img / 255.0 #normalize pixel values
    return img

# load the labels into a dataframe
def load_labels():
    print("Loading labels…")
    # read data into a pandas dataframe
    csv_path = 'labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
    
    # replace NaN values with 0s
    labels.fillna(0, inplace=True)
    
    print("Done.\n")
    return labels

# load the images and labels of eyes 
def load_eyes():
    # load labels
    labels = load_labels()
    y_eyes = labels.iloc [:, 2]
    x_eyes = []
    check = []
    
    print("Loading eyes…")
    # set up the path
    eyes_path = 'data/eyes'
    eyes_dir = os.listdir(eyes_path)
    # iterate over images
    for image in eyes_dir:
        
        check.append(image)
        n = len(check)
        for i in range (n-1):
            if image == check[i]:
                print("ALERT FOR", image)
        
        # load the image
        image_path = os.path.join(eyes_path, image) # changed eyes_dir to eyes_path
        img = cv2.imread(image_path)
        
        # preprocess the image
        if (img is not None):
            x_eyes.append(preprocess_image(img))

    x_eyes = np.array(x_eyes)
    x_eyes = x_eyes.reshape(-1, INPUT_SHAPE_X, INPUT_SHAPE_Y, 1)
    y_eyes = np.array(y_eyes)
    
    print("Done.\n")
    return x_eyes, y_eyes
    
# split the data into train, validation and evaluation sets
def split_data(x_data, y_data):
    print("Splitting data…")
    # split the dataset into train and test sets
    x_train_val, x_test, y_train_val, y_test = model_selection.train_test_split(
        x_data, y_data, test_size=0.2, random_state=42)
    
    # split the train and validation sets
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=42)
    
    
    # make sure all values are integers
    x_train = x_train.astype('int32')
    y_train = y_train.astype('int32')
    x_val = x_val.astype('int32')
    y_val = y_val.astype('int32')
    x_test = x_test.astype('int32')
    y_test = y_test.astype('int32')

    '''
    # one-hot code labels
    y_train = tf.one_hot(y_train,3)
    y_val = tf.one_hot(y_val,3)
    y_test = tf.one_hot(y_test,3)
    '''
    
    print("Done.\n")
    return x_train, y_train, x_val, y_val, x_test, y_test
        
### BUILING & EVALUATING THE MODELS
def create_model():
    print("Creating the model...") 
    model = tf.keras.models.Sequential([
        # convolution layers
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(INPUT_SHAPE_X,INPUT_SHAPE_Y,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        
        # dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    print("Done.\n")   
    return model

# train the model
def train_model(images, labels, val_images, val_labels):
    model = create_model()
    
    # compile the model
    print("Compiling the model...")   
    model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
    
    
    print("Done.\n")   
    
    # train the model
    print("Training the model...")
    history = model.fit(
        images, labels,
        epochs=EPOCHS,
        batch_size = BATCH_SIZE,
        validation_data=(val_images, val_labels),)
    print("Done.\n")
    
    # save the model
    path = 'models-pain/model_' + ITERATION
    model.save(path)
    
    return history, model

# evaluate the model
def evaluate_model(model, images, labels):
    print("Evaluating the model.") 
    results = model.evaluate(images, labels)
    print("Done.\n") 
    return results

# loads the most recent saved model
def load_model():
    print("Loading model…")
    path = 'models-pain/model_' + ITERATION
    model = tf.keras.models.load_model(path)
    print("Done.\n")
    return model

### SCORING EACH CATEGORY   
def eyes():
    # get the data
    x_eyes, y_eyes = load_eyes()
    print(np.shape(x_eyes), np.shape(y_eyes))
    '''
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_eyes, y_eyes)

    # train the model
    history, model = train_model(x_train, y_train, x_val, y_val)
    print(history)
    
    # evaluate the model
    results = evaluate_model(model, x_test, y_test)
    print(results)
    '''

### MAIN
def main():
    eyes()

main()