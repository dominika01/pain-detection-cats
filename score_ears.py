
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# global variables
INPUT_SHAPE_X = 128
INPUT_SHAPE_Y = 64
ITERATION = '1'
EPOCHS = 50
BATCH_SIZE = 1024

### DATA PREPROCESSING

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
    i = 0
    
    # iterate over images
    for image in ears_dir:
        # load the image
        image_path = os.path.join(ears_path, image)
        img = cv2.imread(image_path)
        
        # preprocess the image
        if (img is not None):
            x_ears.append(preprocess_image(img))
    x_ears = np.array(x_ears)
    x_ears = x_ears.reshape(-1, INPUT_SHAPE_X, INPUT_SHAPE_Y, 1)

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
        
### BUILING & EVALUATING THE MODEL
def create_model():
    print("Creating the model...") 
    model = tf.keras.models.Sequential([
        # convolution layers
        tf.keras.layers.Conv2D(32, (3,3), activation='relu',  padding='same', 
                               input_shape=(INPUT_SHAPE_X,INPUT_SHAPE_Y,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),

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
def train_model(images, labels, val_images, val_labels, weights):
    model = create_model()
    
    # compile the model
    print("Compiling the model...")   
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
    
    
    print("Done.\n")   
    
    # train the model
    print("Training the model...")
    history = model.fit(
        images, labels,
        epochs=EPOCHS,
        batch_size = BATCH_SIZE,
        validation_data=(val_images, val_labels),
        #class_weight=weights
        )
    print("Done.\n")
    
    # save the model
    path = 'models-pain/model_' + ITERATION
    model.save(path)
    
    return history, model

# define function to create the model
def create_model_w():
    print("Creating the model...")
    model = tf.keras.models.Sequential([
        # convolution layers
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(INPUT_SHAPE_X,INPUT_SHAPE_Y,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        
        # dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    print("Done.\n")
    
    # compile the model
    print("Compiling the model...")
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
    print("Done.\n")
    
    return model

def tuning_weights():
    # load data
    x_ears, y_ears = load_ears()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_ears, y_ears)
    
    # init search       
    param_grid = []
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model_w, epochs=10, batch_size=32, verbose=0)
    random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42)
    random_search.fit(x_train, y_train)

    # Print the best score and parameters
    print("Best score: %f" % (random_search.best_score_))
    print("Best parameters: %s" % (random_search.best_params_))
    
    best_model = random_search.best_estimator_
    # evaluate the model
    results = evaluate_model(best_model, x_test, y_test)
    print(results)
    
    confusion_matrix(best_model,x_test,y_test)
    
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

def ears():
    # get the data
    x_ears, y_ears = load_ears()
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_ears, y_ears)

    # train the model
    #weights = {0: 1., 1: 1.8, 2: 5.}
    weights = {0: 1., 1: 6, 2: 10}
    history, model = train_model(x_train, y_train, x_val, y_val, weights)
    print(history)
    
    # evaluate the model
    results = evaluate_model(model, x_test, y_test)
    print(results)
    
    confusion_matrix(model,x_test,y_test)
    
## TRANSFER LEARNING
def ears_vgg16():
    # iterate over images
    ears_path = 'data/ears'
    ears_dir = os.listdir(ears_path)
    x_ears = []
    for image in ears_dir:
        image_path = os.path.join(ears_path, image)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 64))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x_ears.append(x)
    
    # load labels
    print(len(x_ears))
    labels = load_labels()
    y_ears = labels.iloc [:, 1]
    y_ears = np.array(y_ears)
    x_ears = tf.keras.applications.vgg16.preprocess_input(np.array(x_ears))
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.applications.vgg16.VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                                                include_top=False,
                                                input_shape=(INPUT_SHAPE_X, INPUT_SHAPE_Y, 3)))
    model.add(tf.keras.layers.Flatten(input_shape=model.output_shape[1:]))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    # Compile the model with appropriate optimizer, loss function, and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on your data
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    results = model.evaluate(x_test, y_test)
    print(results)
    
    x_ears, y_ears = load_ears()
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_ears, y_ears)
    model=load_model()
    confusion_matrix(model,x_test,y_test)

def ears_resnet():
    # iterate over images
    ears_path = 'data/ears'
    ears_dir = os.listdir(ears_path)
    x_ears = []
    for image in ears_dir:
        image_path = os.path.join(ears_path, image)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 64))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x_ears.append(x)
    
    # load labels
    print(len(x_ears))
    labels = load_labels()
    y_ears = labels.iloc [:, 1]
    y_ears = np.array(y_ears)
    x_ears = tf.keras.applications.vgg16.preprocess_input(np.array(x_ears))
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_ears, y_ears)
    
    # setup
    pretrained_model = tf.keras.applications.resnet50.ResNet50(input_shape=(INPUT_SHAPE_X, INPUT_SHAPE_Y, 3),
                                                              include_top = False,
                                                              weights = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    # add new layers    
    x = tf.keras.layers.Flatten()(pretrained_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    # combine
    model = tf.keras.models.Model(pretrained_model.input, x)
    
    # Compile the model with appropriate optimizer, loss function, and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on your data
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    results = model.evaluate(x_test, y_test)
    print(results)
    
    confusion_matrix(model,x_test,y_test)

def ears_inception():
    # iterate over images
    ears_path = 'data/ears'
    ears_dir = os.listdir(ears_path)
    x_ears = []
    for image in ears_dir:
        image_path = os.path.join(ears_path, image)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(INPUT_SHAPE_X, INPUT_SHAPE_Y))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x_ears.append(x)
    
    # load labels
    print(len(x_ears))
    labels = load_labels()
    y_ears = labels.iloc [:, 1]
    y_ears = np.array(y_ears)
    x_ears = tf.keras.applications.vgg16.preprocess_input(np.array(x_ears))
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_ears, y_ears)
    
    # setup
    pretrained_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(INPUT_SHAPE_X, INPUT_SHAPE_Y, 3),
                                                              include_top = False,
                                                              weights = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    # add new layers
    x = tf.keras.layers.Flatten()(pretrained_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    # combine
    model = tf.keras.models.Model(pretrained_model.input, x)
    
    # Compile the model with appropriate optimizer, loss function, and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on your data
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    results = model.evaluate(x_test, y_test)
    print(results)
    
    confusion_matrix(model,x_test,y_test)

def confusion_matrix(model, x_test, y_test):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
    from sklearn.metrics import recall_score, f1_score, classification_report
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred,0)
    y_pred = y_pred.astype(int)
    
    confusion = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(y_pred).argmax(axis=1))
    print('Confusion Matrix\n')
    print(confusion)

    #importing accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['Score 0', 'Score 1', 'Score 2']))
    
    import seaborn as sns

    lables = ['0','1','2']    

    ax= plt.subplot()

    sns.heatmap(confusion, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(lables); ax.yaxis.set_ticklabels(lables);
    plt.show()
  
### MAIN
def main():
    ears()
    #ears_vgg16()
    #ears_resnet()
    #ears_inception()
    #tuning_weights()
    

main()