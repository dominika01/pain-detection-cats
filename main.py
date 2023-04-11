import cv2
import numpy as np
import tensorflow as tf
import keypoints
import joblib

# path to the image
print("Loading the image…")
image_path = 'data-keypoints/CAT_00/00000001_029.jpg' # change to cmd line arg later?
image = cv2.imread(image_path)

# crop the features
print("Cropping the features…")
ears, eyes, muzzle = keypoints.predict_and_crop(image_path)
feature_scores = []

# classify ears
print("\nClassifying ears…")
ears_resized = cv2.resize(ears, (128, 64))
ears_preprocessed = tf.keras.preprocessing.image.img_to_array(ears_resized)
ears_preprocessed = ears_preprocessed.reshape(1, 128, 64, 3)
model_ears = tf.keras.models.load_model('model-ears')
ears_probability_array = model_ears.predict(ears_preprocessed)
ears_score = np.argmax(ears_probability_array, axis=1)
ears_score = ears_score[0]
ears_confidence = round(np.max(ears_probability_array), 2)
print("Score:", ears_score)
print("Confidence:", ears_confidence)
feature_scores.append(ears_score)

# classify eyes
print("\nClassifying eyes…")
eyes_resized = cv2.resize(eyes, (128, 64))
eyes_preprocessed = tf.keras.preprocessing.image.img_to_array(eyes_resized)
eyes_preprocessed = eyes_preprocessed.reshape(1, 128, 64, 3)
model_eyes = tf.keras.models.load_model('model-eyes')
eyes_probability_array = model_eyes.predict(eyes_preprocessed)
eyes_score = np.argmax(eyes_probability_array, axis=1)
eyes_score = eyes_score[0]
eyes_confidence = round(np.max(eyes_probability_array), 2)
print("Score:", eyes_score)
print("Confidence:", eyes_confidence)
feature_scores.append(ears_score)

# classify muzzle
print("\nClassifying muzzle…")
muzzle_resized = cv2.resize(muzzle, (128, 64))
muzzle_preprocessed = tf.keras.preprocessing.image.img_to_array(muzzle_resized)
muzzle_preprocessed = muzzle_preprocessed.reshape(1, 128, 64, 3)
model_muzzle = tf.keras.models.load_model('model-muzzle')
muzzle_probability_array = model_muzzle.predict(muzzle_preprocessed)
muzzle_score = np.argmax(muzzle_probability_array, axis=1)
muzzle_score = muzzle_score[0]
muzzle_confidence = round(np.max(muzzle_probability_array), 2)
print("Score: ", muzzle_score)
print("Confidence: ", muzzle_confidence)

# classify whiskers
print("\nClassifying whiskers…")
whiskers_resized = cv2.resize(muzzle, (128, 64))
whiskers_preprocessed = tf.keras.preprocessing.image.img_to_array(whiskers_resized)
whiskers_preprocessed = whiskers_preprocessed.reshape(1, 128, 64, 3)
model_whiskers = tf.keras.models.load_model('model-whiskers')
whiskers_probability_array = model_whiskers.predict(whiskers_preprocessed)
whiskers_score = np.argmax(whiskers_probability_array, axis=1)
whiskers_score = whiskers_score[0]
whiskers_confidence = round(np.max(whiskers_probability_array), 2)
print("Score: ", whiskers_score)
print("Confidence: ", whiskers_confidence)

# classify head
print("\nClassifying head…")
head_resized = cv2.resize(image, (256, 256))
head_preprocessed = tf.keras.preprocessing.image.img_to_array(head_resized)
head_preprocessed = head_preprocessed.reshape(1, 256, 256, 3)
model_head = tf.keras.models.load_model('model-head')
head_probability_array = model_head.predict(head_preprocessed)
head_score = np.argmax(head_probability_array, axis=1)
head_score = head_score[0]
head_confidence = round(np.max(head_probability_array), 2)
print("Score: ", head_score)
print("Confidence: ", head_confidence)

# calculate score
feature_scores = np.array(feature_scores)
feature_scores = feature_scores.reshape((1, 5))
model_score = joblib.load('model-score.joblib')
result = model_score.predict(feature_scores)
result = np.round(result).astype(int)
final_score = result[0]

# display results
print("\n\nResults\n")
print("Ear position\t", ears_score)
print("Orbital tightening\t", eyes_score)
print("Muzzle tension\t", muzzle_score)
print("Whiskers position\t", whiskers_score)
print("Head position\t", head_score)
print("\nOverall impression\n", final_score)