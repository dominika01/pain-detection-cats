import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
import tensorflow as tf
from keras.metrics import RootMeanSquaredError, MeanSquaredLogarithmicError
import tensorflow.keras.backend as K
from sklearn import model_selection

def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_res/(SS_tot + K.epsilon())

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
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return img

# preprocess an individual set of keypoints
def preprocess_keypoints(kp, img):
    kp = kp.reshape(-1, 2)
    kp = kp * [256 / img.shape[1], 256 / img.shape[0]]
    kp = kp / 255.0
    return kp

# load the images and their corresponding key points
def load_data():
    folder_dir, folders = setup_paths()
    images = []
    keypoints = []
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
    print("Pre-processed data.")    
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
    
    # split the dataset into train and test sets
    train_val_images, test_images, train_val_keypoints, test_keypoints = model_selection.train_test_split(
        images, keypoints, test_size=0.2, random_state=42)

    # split the train and validation sets
    train_images, val_images, train_keypoints, val_keypoints = model_selection.train_test_split(
        train_val_images, train_val_keypoints, test_size=0.2, random_state=42)
    
    print("Split data into sets.")   
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
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(18)
    ])
    print("Created the model.")   
    return model

# train the model
def train_model(train_images, train_keypoints, val_images, val_keypoints):
    model = create_model()
    model.compile(optimizer='adam', loss="mean_squared_error", 
                  metrics=['mae', RootMeanSquaredError(name='rmse'), 
                           MeanSquaredLogarithmicError(name='msle'), r2])
    history = model.fit(train_images, train_keypoints, 
                        epochs=9, batch_size=32,
                        validation_data=(val_images, val_keypoints))
    model.save('models/model_1')
    print("Trained the model.")   
    return history, model

# evaluate the model
def evaluate_model(model, test_images, test_keypoints):
    results = model.evaluate(test_images, test_keypoints)
    print("Evaluated the model.")   
    return results

# predict and display a given number of images with their keypoints
def predict_and_display(model, test_images, test_keypoints, n):
    if n<=0:
        return
    
    print("Making predictions.")   
    pred = model.predict(test_images)
    print("Made predictions. Displaying.")   
    
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
    
def predict_from_path (model, path, n):
    if n<=0:
        return
    
    # preprocess images
    images = []
    
    for image_path in path:
            # load image
            img = cv2.imread(image_path) # apparently this is null
            images.append(preprocess_image(img))
    
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)
    
    for i in range (n):
        # predict
        print("Making predictions.")   
        pred = model.predict(images[i])
        print("Made predictions. Displaying.")   
        
        # undo data pre-processing data
        img = np.squeeze(images[i])

        pr = pred[i]*255
        pr = pr.reshape(-1, 2)
        
        # display
        plt.imshow(img)
        plt.plot(*zip(*pr), marker='o', color='r', ls='')
        plt.show()
     
# run the code
def main():
    
    # training
    train_images, train_keypoints, val_images, val_keypoints, test_images, test_keypoints = prepare_data()
    history, model = train_model(train_images, train_keypoints, val_images, val_keypoints)
    print(history)
    
    # evaluation
    results = evaluate_model(model, test_images, test_keypoints)
    print("Results: ", results)
    
    # making predictions
    #model = tf.keras.models.load_model("models/model_1")
    #predict_and_display(model, test_images, test_keypoints, 5)
    #predict_from_path(model, "data-pain", 10)
    
main()