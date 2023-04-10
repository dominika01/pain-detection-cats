import cv2
import numpy as np
import tensorflow as tf
import keypoints


# path to the image
print("Loading the image…")
image_path = 'data-keypoints/CAT_00/00000001_029.jpg' # change to cmd line arg later?
image = cv2.imread(image_path)

# crop the features
print("Cropping the features…")
ears, eyes, muzzle = keypoints.predict_and_crop(image_path)


#### NEED TO BUILD THESE MODELS ON MY LAPTOP CAUSE THE COLAB VERSION IS HIGHER AND THEY DON'T LOAD
# classify eyes
print("Classifying eyes…")
eyes_resized = cv2.resize(eyes, (128, 64))
eyes_preprocessed = tf.keras.preprocessing.image.img_to_array(eyes_resized)
model_eyes = tf.keras.models.load_model('results/eyes-results/model')
eyes_probability_array = model_eyes.predict(eyes_preprocessed)
eyes_score = np.argmax(eyes_probability_array, axis=1)
print(eyes_score)

# classify ears
print("Classifying ears…")
ears_resized = cv2.resize(ears, (128, 64))
ears_preprocessed = tf.keras.preprocessing.image.img_to_array(ears_resized)
model_ears = tf.keras.models.load_model('results/ears-results/model')
ears_probability_array = model_ears.predict(ears_preprocessed)
ears_score = np.argmax(ears_probability_array, axis=1)
print(ears_score)

# classify muzzle

# classify whiskers

# classify head

# calculate score