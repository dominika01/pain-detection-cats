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
print(np.shape(eyes),np.shape(eyes_resized),np.shape(eyes_preprocessed))
model_eyes = tf.keras.models.load_model('model-eyes') # there's some shape issue???
eyes_probability_array = model_eyes.predict(eyes_preprocessed)
eyes_score = np.argmax(eyes_probability_array, axis=1)
print(eyes_score)

# classify ears

# classify muzzle

# classify whiskers

# classify head

# calculate score