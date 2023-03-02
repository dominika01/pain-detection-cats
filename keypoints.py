import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
import tensorflow as tf
from keras.metrics import RootMeanSquaredError
from sklearn import model_selection

# global variables
INPUT_SHAPE = 256
ITERATION = '11'
EPOCHS = 32
BATCH_SIZE = 32

### EVALUATION FUNCTIONS
# r2 function for evaluation
def r2(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred)) 
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true))) 
    return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())

# euclidean distance function for evaluation
def euclidean_distance(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))

# mean euclidean distance function for evaluation
def med(y_true, y_pred):
    return tf.reduce_mean(euclidean_distance(y_true, y_pred))

# percentage correct keypoints function for evaluation
def pck(y_true, y_pred):
    threshold=0.1
    distance = euclidean_distance(y_true, y_pred)
    pck = tf.cast(distance <= threshold, tf.float32)
    return tf.reduce_mean(pck)


### DATA PREPROCESSING
# set up paths to folders
def setup_paths():
    folder_dir = "data-keypoints"
    folders = ['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03',
               'CAT_04', 'CAT_05', 'CAT_06']
    return folder_dir, folders

# preprocess an individual image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray
    img = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE)) #resize
    img = img / 255.0 #normalize pixel values
    return img

# preprocess an individual set of keypoints
def preprocess_keypoints(kp, img):
    kp = kp.reshape(-1, 2) #reshape the array
    kp = kp * [INPUT_SHAPE / img.shape[1], INPUT_SHAPE / img.shape[0]] #resize
    kp = kp / 255.0 # normalize values
    return kp

# load the images and their corresponding key points
def load_data():
    # set up
    folder_dir, folders = setup_paths()
    images = []
    keypoints = []
    print("Loading data…")
    
    # iterate over the directories
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


### BUILING & EVALUATING THE MODEL
# define the CNN architecture - might improve later
def create_model():
    print("Creating the model...") 
    model = tf.keras.Sequential([
        # convolution layers
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
        
        # dense layers
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
                  metrics=['mae', RootMeanSquaredError(name='rmse'), r2, med, pck])
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


### PREDICTING
# predict and display a given number of test images with their keypoints
def predict_and_display(model, test_images, test_keypoints, n):
    if n<=0:
        return
    
    # make predictions
    print("Making predictions…")   
    pred = model.predict(test_images)
    print("Done.\n")   
    
    # display predictions
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
def predict(path):
    # load model
    model = load_model()
    
    # read image
    img = cv2.imread(path)
    
    # preprocess image
    new_img = []
    new_img.append(preprocess_image(img))
    new_img = np.array(new_img)
    new_img = np.expand_dims(new_img, axis=-1)
    
    # predict
    print("Making predictions…")   
    pred = model.predict(new_img)
    print("Done.\n")   
    
    # reshape original image
    img = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE))

    # undo keypoint preprocessing
    pr = pred*255
    pr = pr.reshape(-1, 2)
    
    # display
    plt.imshow(img)
    plt.plot(*zip(*pr), marker='o', color='r', ls='')
    plt.show()
    
    crop(img, pr)
    
# loads the most recent saved model
def load_model():
    print("Loading model…")
    path = 'models/model_' + ITERATION
    model = tf.keras.models.load_model(
            path, custom_objects={'r2':r2, 'med':med, 'pck':pck})
    print("Done.\n")
    return model

# crop an image according to its keypoints
def crop(img, kp):
    # split kp into x and y
    x = kp[:, 0]
    y = kp[:, 1]
    
    # find x extremities
    x_min = int(np.amin(x))-10
    x_max = int(np.amax(x))+10
    
    # find the y extremities of ears
    y_ears = y[3:]
    y_min_ears = int(np.amin(y_ears))-10
    y_max_ears = int(np.amax(y_ears))+10
    
    # find the nose y
    y_nose = int(y[2])
    
    # find average eye and ear y
    y_eyes = (y[0] + y[1]) / 2
    y_ears_average = int(np.average(y_ears))
    eye_ear_distance = int(y_ears_average) - int(y_eyes)
    
    # find the y_min of the muzzle
    y_muzzle = y_nose - eye_ear_distance

    # crop ears
    ears = img[y_min_ears:y_max_ears, x_min:x_max]
    #cv2.imwrite("ears.jpg", ears)
    cv2.imshow("ears", ears)
    cv2.waitKey(0)
    
    
    # crop eyes
    y_max_ears -= 20
    eyes = img[y_max_ears:y_nose, x_min:x_max]
    #cv2.imwrite("eyes.jpg", eyes)
    cv2.imshow("eyes", eyes)
    cv2.waitKey(0)
    
    # crop muzzle
    y_nose -= 10
    muzzle = img[y_nose:y_muzzle, x_min:x_max]
    #cv2.imwrite("muzzle.jpg", muzzle)
    cv2.imshow("muzzle", muzzle)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

### MAIN
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
    
    # show predictions for n test images
    n = 100
    predict_and_display(model, test_images, test_keypoints, n)
    
#main()   
predict('data-pain/0a0b0c12-52db-40a9-9cf0-00d3805687aa.jpeg')