import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

### UNDERSAMPLING APPROACH

# global variables
INPUT_SHAPE_X = 128
INPUT_SHAPE_Y = 64
ITERATION = '4'
EPOCHS = 250
BATCH_SIZE = 8

### NEXT IDEA
# - add dropout or L2 reg because the model might have converged

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
    csv_path = 'data-labels/labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
    
    # replace NaN values with 0s
    labels.fillna(0, inplace=True)
    
    print("Done.\n")
    return labels
  
# load the images and labels of eyes  
def load_eyes():
    # load labels
    labels = load_labels()
    
    print("Loading eyes…")
    # set up the path
    eyes_path = 'data/eyes'
    eyes_dir = os.listdir(eyes_path)
    
    # init arrays
    x_eyes = []
    y_eyes = []
    
    # for undersampling purposes
    max_images = 1000 # count of minority class
    class_0 = 0
    class_1 = 0
    class_2 = 0
    i = 0
    
    # iterate over images
    for image in eyes_dir:
        # load the image
        image_path = os.path.join(eyes_path, image)
        img = cv2.imread(image_path)
        
        # preprocess the image
        if (img is not None):
            img = preprocess_image(img)
        
            # check image class
            image_class = labels.loc[labels['imageid'] == image, 'orbital_tightening']
    
            if not image_class.empty:
                image_class = image_class.iloc[0]    
                
                # append an equal number of images from each class
                if (image_class == 0.0 and class_0 < max_images):
                    x_eyes.append(img)
                    y_eyes.append(image_class)
                    class_0 += 1
                    if (i == max_images*3):
                        break
                    i+=1
                    
                    
                elif (image_class == 1.0 and class_1 < max_images):
                    x_eyes.append(img)
                    y_eyes.append(image_class)
                    class_1 += 1
                    if (i == max_images*3):
                        break
                    i+=1
                    
                    
                elif (image_class == 2.0 and class_2 < max_images):
                    x_eyes.append(img)
                    y_eyes.append(image_class)
                    class_2 += 1
                    if (i == max_images*3):
                        break
                    i+=1
                
    x_eyes = np.array(x_eyes)
    y_eyes = np.array(y_eyes)
    x_eyes = x_eyes.reshape(-1, INPUT_SHAPE_X, INPUT_SHAPE_Y, 1)
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
        tf.keras.layers.Conv2D(32, (3,3), activation=tf.keras.layers.LeakyReLU(),  padding='same', 
                               input_shape=(INPUT_SHAPE_X,INPUT_SHAPE_Y,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Conv2D(64, (3,3), activation=tf.keras.layers.LeakyReLU(), padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Conv2D(128, (3,3), activation=tf.keras.layers.LeakyReLU(), padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        

        tf.keras.layers.Flatten(),
        
        # dense layers
        tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU()),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    print("Done.\n")   
    return model

# train the model
def train_model(images, labels, val_images, val_labels, weights):
    model = create_model()
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
                                                                decay_steps=1000,
                                                                decay_rate=0.9)
    
    # compile the model
    print("Compiling the model...")   
    model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-7, momentum=0.9),
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

def conf_matrix(model, x_test, y_test):
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    
    prob_array = model.predict(x_test)
    class_indices = np.argmax(prob_array, axis=1)
    print(prob_array)
    # Convert the class indices to a one-hot encoded array
    class_indices = np.argmax(prob_array, axis=1)
    num_classes = 3
    y_pred = np.zeros((prob_array.shape[0], num_classes))
    y_pred[np.arange(prob_array.shape[0]), class_indices] = 1
    
    print(y_pred)
    print(np.shape(y_pred))
    
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

def save_history(hist):
    history_df = pd.DataFrame(hist.history)
    path = 'history/eyes_history_' + ITERATION +'.csv'
    history_df.to_csv(path, index=False)
    
def eyes():
    # get the data
    x_eyes, y_eyes = load_eyes()
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_eyes, y_eyes)

    # train the model
    weights = {0: 1., 1: 1., 2: 1}
    history, model = train_model(x_train, y_train, x_val, y_val, weights)
    print(history)
    with open('hist.txt', 'w') as f:
        # Print to the file using the file parameter
        print(history, file=f)
    
    # evaluate the model
    results = evaluate_model(model, x_test, y_test)
    print(results)
    
    conf_matrix(model,x_test,y_test)

def eyes_vgg16():
    # iterate over images
    eyes_path = 'data/eyes'
    eyes_dir = os.listdir(eyes_path)
    labels = load_labels()
    x_eyes = []
    y_eyes = []
    max_images = 1374
    i = 0
    class_0 = 0
    class_1 = 0
    class_2 = 0
    
    # iterate over images
    for image in eyes_dir:
        # load the image
        image_path = os.path.join(eyes_path, image)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(INPUT_SHAPE_X, INPUT_SHAPE_Y))
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        # preprocess the image
        if (img is not None):
        
            # check image class
            image_class = labels.loc[labels['imageid'] == image, 'orbital_tightening']
    
            if not image_class.empty:
                image_class = image_class.iloc[0]    
                
                # append an equal number of images from each class
                if (image_class == 0.0 and class_0 < max_images):
                    x_eyes.append(img)
                    y_eyes.append(image_class)
                    class_0 += 1
                    if (i == max_images*3):
                        break
                    i+=1
                    
                    
                elif (image_class == 1.0 and class_1 < max_images):
                    x_eyes.append(img)
                    y_eyes.append(image_class)
                    class_1 += 1
                    if (i == max_images*3):
                        break
                    i+=1
                    
                    
                elif (image_class == 2.0 and class_2 < max_images):
                    x_eyes.append(img)
                    y_eyes.append(image_class)
                    class_2 += 1
                    if (i == max_images*3):
                        break
                    i+=1
    
    # add images from class 2
    flipped_path = 'data/ears-flipped'
    for image in os.listdir(flipped_path):
        image_path = os.path.join(flipped_path, image)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(INPUT_SHAPE_X, INPUT_SHAPE_Y))
        img = tf.keras.preprocessing.image.img_to_array(img)
        x_eyes.append(img)
        y_eyes.append(2)
        class_2 += 1
    
    print(class_0, class_1, class_2)
    # load labels
    x_eyes = tf.keras.applications.vgg16.preprocess_input(np.array(x_eyes))
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_eyes, y_eyes)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.applications.vgg16.VGG16(weights='model-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                                                include_top=False,
                                                input_shape=(INPUT_SHAPE_X, INPUT_SHAPE_Y, 3)))
    model.add(tf.keras.layers.Flatten(input_shape=model.output_shape[1:]))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    # Compile the model with appropriate optimizer, loss function, and metrics
    earlystop = EarlyStopping(monitor='val_loss', patience=5)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model on your data
    hist = model.fit(x_train, y_train, 
              epochs=EPOCHS,
              batch_size=BATCH_SIZE, 
              validation_data=(x_val, y_val),
              callbacks=[earlystop])
    
    # save the model
    path = 'models-pain/model_' + ITERATION
    model.save(path)
    
    # evaluate
    results = model.evaluate(x_test, y_test)
    print(results)
    try:
        save_history(hist)
    except:
        print("couldn't save history")
    
    conf_matrix(model,x_test,y_test)
    
### MAIN
def main():
    #eyes()
    eyes_vgg16()

main()
