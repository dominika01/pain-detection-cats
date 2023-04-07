import cv2
import numpy as np
import tensorflow as tf
import keypoints


# path to the image
image_path = 'insert/image/path' # change to cmd line arg later?
image = cv2.imread(image_path)

# crop the features
image_preprocessed = keypoints.preprocess_image(image)
model_kp = keypoints.load_model()
kp = model_kp.predict(image)
ears, eyes, muzzle = keypoints.crop(image, kp)

# classify eyes
eyes_resized = cv2.resize(eyes, (128, 64))
eyes_preprocessed = tf.keras.preprocessing.image.img_to_array(eyes_resized)
model_eyes = tf.keras.models.load_model('eyes_results/model')
eyes_probability_array = model_eyes.predict(eyes_preprocessed)
eyes_score = np.argmax(eyes_probability_array, axis=1)

# classify ears
ears_resized = cv2.resize(ears, (128, 64))
ears_preprocessed = tf.keras.preprocessing.image.img_to_array(ears_resized)
model_ears = tf.keras.models.load_model('ears_results/model')
ears_probability_array = model_ears.predict(ears_preprocessed)
ears_score = np.argmax(ears_probability_array, axis=1)

# classify muzzle

# classify whiskers

# classify head

# calculate score