
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

# own model - 0.5167 with 8 epochs
# vgg16 - 0.5223 with 10 epochs, 3 layers and cross enntropy loss function, to run with MSE
# resnet - 

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

# load the images and labels of muzzle 
def load_muzzle():
    # load labels
    labels = load_labels()
    y_muzzle = labels.iloc [:, 3]
    x_muzzle = []
    
    print("Loading muzzle…")
    # set up the path
    muzzle_path = 'data/muzzle'
    muzzle_dir = os.listdir(muzzle_path)
    # iterate over images
    for image in muzzle_dir:
        # load the image
        image_path = os.path.join(muzzle_path, image) # changed muzzle_dir to muzzle_path
        img = cv2.imread(image_path)
        
        # preprocess the image
        if (img is not None):
            x_muzzle.append(preprocess_image(img))

    x_muzzle = np.array(x_muzzle)
    x_muzzle = x_muzzle.reshape(-1, INPUT_SHAPE_X, INPUT_SHAPE_Y, 1)
    y_muzzle = np.array(y_muzzle)
    
    print("Done.\n")
    return x_muzzle, y_muzzle
    
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
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(INPUT_SHAPE_X,INPUT_SHAPE_Y,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        
        # dense layers
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
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

### TRANSFER LEARNING ATTEMPTS
def vgg16_model():
    # Create a new model with the pre-trained VGG16 as the base and your own fully connected layers on top
    model = tf.keras.models.Sequential()
    model.add(tf.keras.applications.vgg16.VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                                                include_top=False,
                                                input_shape=(INPUT_SHAPE_X, INPUT_SHAPE_Y, 3)))
    model.add(tf.keras.layers.Flatten(input_shape=model.output_shape[1:]))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

    # Compile the model with appropriate optimizer, loss function, and metrics
    muzzle_path = 'data/muzzle'
    muzzle_dir = os.listdir(muzzle_path)
    x_muzzle = []
    
    for image in muzzle_dir:
        image_path = os.path.join(muzzle_path, image)
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, 
                                                        target_size=(INPUT_SHAPE_X, INPUT_SHAPE_Y))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x_muzzle.append(x)
        except:
            print("Couldn't load", image)
    
    # load labels
    print(len(x_muzzle))
    labels = load_labels()
    y_muzzle = labels.iloc [:, 1]
    y_muzzle = np.array(y_muzzle)
    x_muzzle = tf.keras.applications.vgg16.preprocess_input(np.array(x_muzzle))
    
    model.compile(optimizer='adam', loss='mse', 
                  metrics=['accuracy'])

    # Train the model on your data
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_muzzle, y_muzzle)
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    results = model.evaluate(x_test, y_test)
    print(results)

def resnet_model():
    # setup
    shape = (INPUT_SHAPE_X, INPUT_SHAPE_Y,1)
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
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    
    # resnet conv block 1
    filter*=2
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
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
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
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
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
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
    
    pass

def muzzle_resnet():
    # get the data
    x_muzzle, y_muzzle = load_muzzle()

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_muzzle, y_muzzle)

    # train the model
    model = create_model()
    
    # compile the model
    print("Compiling the model...")   
    model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
    
    
    print("Done.\n")   
    
    # train the model
    print("Training the model...")
    history, model = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size = BATCH_SIZE,
        validation_data=(x_val, y_val),)
    print("Done.\n")
    
    # evaluate the model
    results = evaluate_model(model, x_test, y_test)
    print(results)

def muzzle():
    # get the data
    x_muzzle, y_muzzle = load_muzzle()

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_muzzle, y_muzzle)

    # train the model
    history, model = train_model(x_train, y_train, x_val, y_val)
    print(history)
    
    # evaluate the model
    results = evaluate_model(model, x_test, y_test)
    print(results)
    

### MAIN
def main():
    #muzzle()
    vgg16_model() #run with the other loss function
    #muzzle_resnet()

main()