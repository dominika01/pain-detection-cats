import cv2
import numpy as np
import tensorflow as tf
import keypoints
import joblib

def preprocess_feature (feature, x_size, y_size):
    resized = cv2.resize(feature, (x_size, y_size))
    preprocessed = tf.keras.preprocessing.image.img_to_array(resized)
    reshaped = preprocessed.reshape(1, x_size, y_size, 3)
    return reshaped

def score_feature (feature, model):
    probability_array = model.predict(feature)
    score = np.argmax(probability_array, axis=1)
    score = score[0]
    confidence = round(np.max(probability_array), 2)
    print("Score:", score)
    print("Confidence:", confidence)
    return score

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
ears_preprocessed = preprocess_feature(ears, 128, 64)
ears_model = tf.keras.models.load_model('model-ears')
ears_score = score_feature (ears_preprocessed, ears_model)
feature_scores.append(ears_score)

# classify eyes
print("\nClassifying eyes…")
eyes_preprocessed = preprocess_feature(eyes, 128, 64)
eyes_model = tf.keras.models.load_model('model-eyes')
eyes_score = score_feature (eyes_preprocessed, eyes_model)
feature_scores.append(eyes_score)

# classify muzzle
print("\nClassifying muzzle…")
muzzle_preprocessed = preprocess_feature(muzzle, 128, 64)
muzzle_model = tf.keras.models.load_model('model-muzzle')
muzzle_score = score_feature (muzzle_preprocessed, muzzle_model)
feature_scores.append(muzzle_score)

# classify whiskers
print("\nClassifying whiskers…")
whiskers_preprocessed = preprocess_feature(muzzle, 128, 64)
whiskers_model = tf.keras.models.load_model('model-whiskers')
whiskers_score = score_feature (whiskers_preprocessed, whiskers_model)
feature_scores.append(whiskers_score)

# classify head
print("\nClassifying head…")
head_preprocessed = preprocess_feature(image, 128, 64)
head_model = tf.keras.models.load_model('model-head')
head_score = score_feature (head_preprocessed, head_model)
feature_scores.append(head_score)

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