import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.regularizers import L2


### GLOBAL VARIABLES
INPUT_SHAPE_X = 128
INPUT_SHAPE_Y = 64
ITERATION = '6'
EPOCHS = 1000
BATCH_SIZE = 8
MAX_IMAGES = 2000 #max images per class
LEARN_RATE = 1e-3


### DATA PREPROCESSING

# load the labels into a dataframe
def load_labels():
    print("Loading labels…")
    # read data into a pandas dataframe
    csv_path = 'data-labels/labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
    
    # replace NaN values with 0s
    labels.fillna(0, inplace=True)
    
    print("Done.\n")
    return labels
  
# load the images and labels of muzzle  
def load_muzzle():
    # load the images
    muzzle_path = 'data/muzzle'
    muzzle_dir = os.listdir(muzzle_path)
    x_muzzle = []
    y_muzzle = []
    i = 0
    class_0 = 0
    class_1 = 0
    class_2 = 0

    # iterate over images
    for image in muzzle_dir:
        # get labels
        labels = load_labels()
        
        # load the image
        image_path = os.path.join(muzzle_path, image)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(INPUT_SHAPE_X, INPUT_SHAPE_Y))
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        # preprocess the image
        if (img is not None):
        
            # check image class
            image_class = labels.loc[labels['imageid'] == image, 'muzzle_tension']

            if not image_class.empty:
                image_class = image_class.iloc[0]    
                
                # append an equal number of images from each class
                if (image_class == 0.0 and class_0 < MAX_IMAGES):
                    x_muzzle.append(img)
                    y_muzzle.append(image_class)
                    class_0 += 1
                    if (i == MAX_IMAGES*3):
                        break
                    i+=1
                    
                    
                elif (image_class == 1.0 and class_1 < MAX_IMAGES-250):
                    x_muzzle.append(img)
                    y_muzzle.append(image_class)
                    class_1 += 1
                    if (i == MAX_IMAGES*3):
                        break
                    i+=1
                    
                    
                elif (image_class == 2.0 and class_2 < MAX_IMAGES):
                    x_muzzle.append(img)
                    y_muzzle.append(image_class)
                    class_2 += 1
                    if (i == MAX_IMAGES*3):
                        break
                    i+=1
    
    # add flipped images from the minority class
    flipped_path = 'data/muzzle-flipped'
    for image in os.listdir(flipped_path):
        image_path = os.path.join(flipped_path, image)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(INPUT_SHAPE_X, INPUT_SHAPE_Y))
        img = tf.keras.preprocessing.image.img_to_array(img)
        x_muzzle.append(img)
        y_muzzle.append(2.0)
        class_2 += 1
        
    # add augmented images from the minority class
    aug_path = 'data/muzzle-augmented'
    for image in os.listdir(aug_path):
        image_path = os.path.join(aug_path, image)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(INPUT_SHAPE_X, INPUT_SHAPE_Y))
        img = tf.keras.preprocessing.image.img_to_array(img)
        x_muzzle.append(img)
        y_muzzle.append(2)
        class_2 += 1
    
    # preprocess the images
    x_muzzle = preprocess_input(np.array(x_muzzle))
    
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
    
    # convert y_train to a NumPy array
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # convert y_train to int32 data type
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    # one-hot code labels
    y_train = tf.one_hot(y_train,3)
    y_val = tf.one_hot(y_val,3)
    y_test = tf.one_hot(y_test,3)
    
    print("Done.\n")
    return x_train, y_train, x_val, y_val, x_test, y_test
        
        
### BUILING THE MODEL

# create a model based on VGG16
def create_model():
    print("Creating the model...") 

    model = tf.keras.models.Sequential()
    model.add(tf.keras.applications.vgg16.VGG16(weights='model-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                                                include_top=False,
                                                input_shape=(INPUT_SHAPE_X, INPUT_SHAPE_Y, 3)))
    model.add(tf.keras.layers.Flatten(input_shape=model.output_shape[1:]))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    
    print("Done.\n") 
    return model
      

# train the model
def train_model(x_train, y_train, x_val, y_val):
    # create the model
    model = create_model()
    
    # compile the model
    print("Compiling the model...")   
    model.compile(optimizer = SGD(learning_rate = LEARN_RATE), 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    print("Done.\n")   
    
    # train the model
    print("Training the model...")
    earlystop = EarlyStopping(monitor = 'val_loss', patience = 10)
    history = model.fit(x_train, y_train, 
                        epochs = EPOCHS,
                        batch_size = BATCH_SIZE, 
                        validation_data = (x_val, y_val),
                        callbacks = [earlystop])
    print("Done.\n")
    
    return history, model


### EVALUATING THE MODEL
# evaluate the model
def evaluate_model(model, x_test, y_test):
    print("Evaluating the model.") 
    results = model.evaluate(x_test, y_test)
    conf_matrix(model, x_test, y_test)
    print("Done.\n") 
    return results

# create a confusion matrix
def conf_matrix(model, x_test, y_test):
    # predict
    prob_array = model.predict(x_test)
    class_indices = np.argmax(prob_array, axis=1)
    print(prob_array)
    
    # convert the class indices to a one-hot encoded array
    class_indices = np.argmax(prob_array, axis=1)
    num_classes = 3
    y_pred = np.zeros((prob_array.shape[0], num_classes))
    y_pred[np.arange(prob_array.shape[0]), class_indices] = 1
    
    print(y_pred)
    print(np.shape(y_pred))
    
    # create the confusion matrix
    confusion = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(y_pred).argmax(axis=1))
    print('Confusion Matrix\n')
    print(confusion)

    # print a classification report
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['Score 0', 'Score 1', 'Score 2']))
    
    # make a plot
    lables = ['0','1','2']    
    ax= plt.subplot()
    sns.heatmap(confusion, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(lables); ax.yaxis.set_ticklabels(lables)
    plt.show()


### SAVING & LOADING

# save training history
def save_history(hist):
    history_df = pd.DataFrame(hist.history)
    path = 'history/muzzle_history_' + ITERATION +'.csv'
    history_df.to_csv(path, index=False)

# save the model
def save_model(model):
    path = 'muzzle_model_' + ITERATION
    model.save(path)
    
# loads the most recent saved model
def load_model():
    print("Loading model…")
    path = 'muzzle_model_' + ITERATION
    model = load_model(path)
    print("Done.\n")
    return model


### MAIN

# run the code 
def main():
    # get and preprocess data
    x_muzzle, y_muzzle = load_muzzle()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_muzzle, y_muzzle)
    
    # train the model
    history, model = train_model(x_train, y_train, x_val, y_val)
    
    # save the data
    save_history(history)
    save_model(model)
    
    # evaluate
    results = evaluate_model(model, x_test, y_test)
    print(results)
    
main()