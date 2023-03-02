import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
import tensorflow as tf
from keras.metrics import RootMeanSquaredError, MeanSquaredLogarithmicError
from sklearn import model_selection

# global variables
INPUT_SHAPE = 256
ITERATION = '10'
EPOCHS = 32
BATCH_SIZE = 32

# Next task: create a more complex CNN architecture

# r2 function for evaluation
def r2(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred)) 
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true))) 
    return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())

def euclidean_distance(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))

def mean_euclidean_distance(y_true, y_pred):
    return tf.reduce_mean(euclidean_distance(y_true, y_pred))

def percentage_correct_keypoints(y_true, y_pred):
    threshold=0.1
    distance = euclidean_distance(y_true, y_pred)
    pck = tf.cast(distance <= threshold, tf.float32)
    return tf.reduce_mean(pck)

# define a function to generate activation maps
def visualize_activation_map(model, input_image, layer_index):
    # create a sub-model that outputs the activations of the specified layer
    activation_model = tf.keras.models.Model(inputs=model.input, 
                                             outputs=model.layers[layer_index].output)
    # generate the activation map for the input image
    activations = activation_model.predict(np.array([input_image]))
    # plot the activation maps as a grid of images
    plt.figure(figsize=(16, 16))
    for i in range(activations.shape[-1]):
        plt.subplot(8, 8, i+1)
        plt.imshow(activations[0, :, :, i], cmap='jet')
        plt.axis('off')
    plt.show()

# set up paths to folders
def setup_paths():
    folder_dir = "data-keypoints"
    folders = ['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03',
               'CAT_04', 'CAT_05', 'CAT_06']
    #folders = ['CAT_00']

    return folder_dir, folders

# preprocess an individual image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE))
    img = img / 255.0
    return img

# preprocess an individual set of keypoints
def preprocess_keypoints(kp, img):
    kp = kp.reshape(-1, 2)
    kp = kp * [INPUT_SHAPE / img.shape[1], INPUT_SHAPE / img.shape[0]]
    kp = kp / 255.0
    return kp

# load the images and their corresponding key points
def load_data():
    folder_dir, folders = setup_paths()
    images = []
    keypoints = []
    print("Loading data…")
    for folder in folders:
        path = os.path.join(folder_dir, folder, '*.jpg')
        images_paths = sorted(glob.glob(path))
        
        for image_path in images_paths:
            # load image
            img = cv2.imread(image_path)
            
            # load keypoints
            with open(image_path+'.cat', 'r') as f:
                keypoints_text = f.read().strip()
            kp = np.array(keypoints_text.split(' ')[1:], dtype=np.float32)
            
            # Add the image and keypoints to the lists
            keypoints.append(preprocess_keypoints(kp, img))
            images.append(preprocess_image(img))
    print("Done.\n")    
    return images, keypoints

# prepare the data and split the sets
def prepare_data():
    # load data
    images, keypoints = load_data()
    
    # convert lists into np arrays
    images = np.array(images)
    keypoints = np.array(keypoints)
    
    # reshape the arrays
    images = np.expand_dims(images, axis=-1)
    keypoints = keypoints.reshape(len(keypoints),18)
    
    print("Splitting data into sets…")   
    
    # split the dataset into train and test sets
    train_val_images, test_images, train_val_keypoints, test_keypoints = model_selection.train_test_split(
        images, keypoints, test_size=0.2, random_state=42)

    # split the train and validation sets
    train_images, val_images, train_keypoints, val_keypoints = model_selection.train_test_split(
        train_val_images, train_val_keypoints, test_size=0.2, random_state=42)
    
    print("Done.\n")   
    return train_images, train_keypoints, val_images, val_keypoints, test_images, test_keypoints

# display an image and its corresponding keypoint
def display_image(images, keypoints, index, original_size):
    # get the image and keypoints
    # and normalise pixel values
    image = images[index] * 255
    keypoints = keypoints[index] * 255
    keypoints = np.reshape(keypoints, (len(images), 9, 2))
    
    # resize the image and its keypoints
    keypoints = keypoints * [original_size[0]/256, original_size[1]/256]
    image = cv2.resize(image, original_size, interpolation=cv2.INTER_LINEAR)

    # display
    plt.imshow(image)
    plt.plot(*zip(*keypoints), marker='o', color='r', ls='')
    plt.show()

# define the CNN architecture - might improve later
def create_model():
    print("Creating the model...") 
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(INPUT_SHAPE, INPUT_SHAPE, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(512, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        
        tf.keras.layers.Dense(18)
    ])
    print("Done.\n")   
    return model

# train the model
def train_model(train_images, train_keypoints, val_images, val_keypoints):
    #create the model
    model = create_model()
    
    # compile the model
    print("Compiling the model...")
    model.compile(optimizer='adam', loss="mean_squared_error", 
                  metrics=['mae', RootMeanSquaredError(name='rmse'), 
                            r2, mean_euclidean_distance, percentage_correct_keypoints])
    print("Done.\n")
    
    # train the model
    print("Training the model...")
    history = model.fit(train_images, train_keypoints, 
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(val_images, val_keypoints))
    print("Done.\n")
    
    # save the model
    path = 'models/model_' + ITERATION
    model.save(path)
    return history, model

# evaluate the model
def evaluate_model(model, test_images, test_keypoints):
    print("Evaluating the model.") 
    results = model.evaluate(test_images, test_keypoints)
    print("Done.\n") 
    return results

# predict and display a given number of test images with their keypoints
def predict_and_display(model, test_images, test_keypoints, n):
    if n<=0:
        return
    
    print("Making predictions…")   
    pred = model.predict(test_images)
    print("Done.\n")   
    
    for i in range (n):
        
        # undo data pre-processing data
        img = np.squeeze(test_images[i])
        
        kp = test_keypoints[i]*255
        kp = kp.reshape(-1, 2)
        
        pr = pred[i]*255
        pr = pr.reshape(-1, 2)
        
        # display
        plt.imshow(img)
        plt.plot(*zip(*kp), marker='o', color='b', ls='')
        plt.plot(*zip(*pr), marker='o', color='r', ls='')
        plt.show()

# predict and display keypoints for a given image
def predict_from_path(model, path):

    img = cv2.imread(path)
    images = []
    images.append(preprocess_image(img))
    
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)
    
    # predict
    print("Making predictions…")   
    pred = model.predict(images)
    print("Done.\n")   
    
    # undo data pre-processing data
    img = np.squeeze(images)

    pr = pred*255
    pr = pr.reshape(-1, 2)
    
    # display
    plt.imshow(img)
    plt.plot(*zip(*pr), marker='o', color='r', ls='')
    plt.show()
    
    visualize_activation_map(model,img,1)
    visualize_activation_map(model,img,3)
    visualize_activation_map(model,img,5)
    visualize_activation_map(model,img,7)
    visualize_activation_map(model,img,9)
    visualize_activation_map(model,img,11)
    visualize_activation_map(model,img,13)
    visualize_activation_map(model,img,15)
    visualize_activation_map(model,img,17)

# loads the most recent saved model
def load_model():
    print("Loading model…")
    path = 'models/model_' + ITERATION
    model = tf.keras.models.load_model(path, custom_objects={'r2':r2})
    print("Done.\n")
    return model

# run the code
def main():
    # pre-processing
    train_images, train_keypoints, val_images, val_keypoints, test_images, test_keypoints = prepare_data()
    
    # training
    history, model = train_model(train_images, train_keypoints, val_images, val_keypoints)
    #print(history)
    
    # evaluation
    results = evaluate_model(model, test_images, test_keypoints)
    print("Results: ", results)
    
    predict_and_display(model, test_images, test_keypoints, 100)
     
def predict():
    # making predictions
    img_path = 'cat.png'
    model = load_model()
    predict_from_path(model, img_path)
    
 
main()   
#predict()