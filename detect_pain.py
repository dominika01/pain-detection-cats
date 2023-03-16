
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn import model_selection

# global variables
INPUT_SHAPE = 256
ITERATION = '1'
EPOCHS = 16
BATCH_SIZE = 32

### DATA PREPROCESSING
# preprocess an individual image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray
    img = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE)) #resize
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
  
# load the images and labels of ears  
def load_ears():
    # load labels
    labels = load_labels()
    y_ears = labels.iloc [:, 1]
    
    print("Loading ears…")
    # set up the path
    ears_path = 'data/ears'
    ears_dir = os.listdir(ears_path)
    x_ears = []
    
    # iterate over images
    for image in ears_dir:
        # load the image
        image_path = os.path.join(ears_path, image)
        img = cv2.imread(image_path)
        # preprocess the image
        x_ears.append(preprocess_image(img))

    x_ears = np.array(x_ears)
    x_ears = x_ears.reshape(-1, INPUT_SHAPE, INPUT_SHAPE, 1)
    
    print("Done.\n")
    return x_ears, y_ears

# load the images and labels of eyes 
def load_eyes():
    print("Loading eyes…")
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
    y_eyes = labels.iloc [:, 2]
    
    print("Done.\n")
    return x_eyes, y_eyes
    
# load the images and labels of muzzles 
def load_muzzle():
    print("Loading muzzles…")
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
    y_muzzle = labels.iloc [:, 3]
    
    print("Done.\n")
    return x_muzzle, y_muzzle

# load the images and labels of whiskers 
def load_whiskers():
    print("Loading whiskers…")
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
    y_whiskers = labels.iloc [:, 4]

# load the images and labels of heads     
def load_head():
    print("Loading heads…")
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
    y_head = labels.iloc [:, 5]
    
    print("Done.\n")
    return x_head, y_head

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

    # one-hot code labels
    y_train = tf.one_hot(y_train,3)
    y_val = tf.one_hot(y_val,3)
    y_test = tf.one_hot(y_test,3)
    print("Done.\n")
    return x_train, y_train, x_val, y_val, x_test, y_test


### BUILING & EVALUATING THE MODELS
def create_model():
    print("Creating the model...") 
    model = tf.keras.models.Sequential([
        # convolution layers
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(INPUT_SHAPE,INPUT_SHAPE,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(1024, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        
        # dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    print("Done.\n")   
    return model

def create_model_resnet():
    num_classes = 3
    input_shape = (INPUT_SHAPE,INPUT_SHAPE,1)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', input_shape=(256, 256, 1))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # ResNet block 1
    shortcut = x
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    # ResNet block 2
    shortcut = x
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
  
    shortcut = tf.keras.layers.Conv2D(128, 1, strides=2, padding='same')(shortcut)
   
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    
    # ResNet block 3
    shortcut = x
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 1, padding='same')(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)


    # ResNet block 4
    shortcut = x
    x = tf.keras.layers.Conv2D(512, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, 1, padding='same')(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def build_model():
    # setup
    shape = (INPUT_SHAPE, INPUT_SHAPE,1)
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    
    # identity block
    filter = 64
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    
    # resnet conv block 1
    filter*=2
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    
    # resnet conv block 2
    filter*=2
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    
    # resnet conv block 3
    filter*=2
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    
    # dense
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(3, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x)
    return model


def train_model(images, labels, val_images, val_labels):
    model = build_model()
    
    # compile the model
    print("Compiling the model...")   
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tfa.metrics.F1Score(num_classes=3)])
    
    
    print("Done.\n")   
    
    # train the model
    print("Training the model...")
    #history = model.fit(images, labels, 
    #                    epochs=EPOCHS, batch_size=BATCH_SIZE,
    #                    validation_data=(val_images, val_labels))
    history = model.fit(
        images, labels,
        epochs=EPOCHS,
        validation_data=(val_images, val_labels))
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
def ears():
    # get the data
    x_ears, y_ears = load_ears()
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_ears, y_ears)
    
    # train the model
    history, model = train_model(x_train, y_train, x_val, y_val)
    print(history)
    
    # evaluate the model
    results = evaluate_model(model, x_test, y_test)
    print(results)
    

def eyes():
    pass

def muzzle():
    pass

def whiskers():
    pass

def head():
    pass


### MAIN
def main():
    ears()
    pass

main()