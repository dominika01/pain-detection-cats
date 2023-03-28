import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import load_img, img_to_array, preprocess_input
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16


### GLOBAL VARIABLES
INPUT_SHAPE_X = 128
INPUT_SHAPE_Y = 64
ITERATION = '1'
EPOCHS = 250
BATCH_SIZE = 8
LEARN_RATE = 1e-6


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
  
# load the images and labels of ears  
def load_ears():
    print("Loading ears…")
    
    # set up the path
    ears_path = 'data/ears'
    ears_dir = os.listdir(ears_path)
    
    # load labels
    labels = load_labels()
    
    # set up arrays
    x_ears = []
    y_ears = []
    
    # define max and class count for undersampling
    max_images = 2377 # number of images in minority class
    i = 0
    class_0 = 0
    class_1 = 0
    class_2 = 0
    
    # iterate over images
    for image in ears_dir:
        # load the image
        image_path = os.path.join(ears_path, image)
        img = load_img(image_path, target_size=(INPUT_SHAPE_X, INPUT_SHAPE_Y))
        img = img_to_array(img)
        
        # preprocess the image
        if (img is not None):
        
            # check image class
            image_class = labels.loc[labels['imageid'] == image, 'ear_position']
    
            if not image_class.empty:
                image_class = image_class.iloc[0]    
                
                # append an equal number of images from each class
                if (image_class == 0.0 and class_0 < max_images):
                    x_ears.append(img)
                    y_ears.append(image_class)
                    class_0 += 1
                    if (i == max_images*3):
                        break
                    i+=1
                    
                    
                elif (image_class == 1.0 and class_1 < max_images):
                    x_ears.append(img)
                    y_ears.append(image_class)
                    class_1 += 1
                    if (i == max_images*3):
                        break
                    i+=1
                    
                    
                elif (image_class == 2.0 and class_2 < max_images):
                    x_ears.append(img)
                    y_ears.append(image_class)
                    class_2 += 1
                    if (i == max_images*3):
                        break
                    i+=1
    
    # add augmented images from the minority class
    flipped_path = 'data/ears-flipped'
    for image in os.listdir(flipped_path):
        image_path = os.path.join(flipped_path, image)
        img = load_img(image_path, target_size=(INPUT_SHAPE_X, INPUT_SHAPE_Y))
        img = img_to_array(img)
        x_ears.append(img)
        y_ears.append(2.0)
        class_2 += 1
    
    # preprocess the images
    x_ears = preprocess_input(np.array(x_ears))
    
    print("Done.\n")
    return x_ears, y_ears
    
# split the data into train, validation and evaluation sets
def split_data(x_data, y_data):
    print("Splitting data…")
    # split the dataset into train and test sets
    x_train_val, x_test, y_train_val, y_test = model_selection.train_test_split(
        x_data, y_data, test_size=0.2, random_state=42)
    
    # split the train and validation sets
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # make sure labels are int
    try:
        y_train = y_train.astype(np.int32)
    except:
        print("issue with train")
        
    try:
        y_test = y_test.astype(np.int32)
    except:
        print("issue with test")
        
    try:
        y_val = y_val.astype(np.int32)
    except:
        print("issue with val")
    
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
    
    model = Sequential()
    model.add(VGG16(weights='model-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                    include_top=False,
                    input_shape=(INPUT_SHAPE_X, INPUT_SHAPE_Y, 3)))
    model.add(Flatten(input_shape=model.output_shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    
    return model
    print("Done.\n")   

# train the model
def train_model(x_train, y_train, x_val, y_val):
    # create the model
    model = create_model()
    
    # compile the model
    print("Compiling the model...")   
    earlystop = EarlyStopping(monitor = 'val_loss', patience = 5)
    model.compile(optimizer = SGD(learning_rate = LEARN_RATE), 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    print("Done.\n")   
    
    # train the model
    print("Training the model...")
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
    path = 'history/ears/ears_history_' + ITERATION +'.csv'
    history_df.to_csv(path, index=False)

# save the model
def save_model(model):
    path = 'models-ears/model_' + ITERATION
    model.save(path)
    
# loads the most recent saved model
def load_model():
    print("Loading model…")
    path = 'models-ears/model_' + ITERATION
    model = load_model(path)
    print("Done.\n")
    return model


### MAIN

# run the code 
def main():
    # get and preprocess data
    x_ears, y_ears = load_ears()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_ears, y_ears)
    
    # train the model
    history, model = train_model(x_train, y_train, x_val, y_val)
    
    # save the data
    save_history(history)
    save_model(model)
    
    # evaluate
    results = model.evaluate(x_test, y_test)
    print(results)
    
main()
