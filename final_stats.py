import cv2
import numpy as np
import tensorflow as tf
import keypoints
import joblib
import pandas as pd
import os
import numpy as np

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

def load_models ():
    ears_model = tf.keras.models.load_model('model-ears')
    eyes_model = tf.keras.models.load_model('model-eyes')
    muzzle_model = tf.keras.models.load_model('model-muzzle')
    whiskers_model = tf.keras.models.load_model('model-whiskers')
    head_model = tf.keras.models.load_model('model-head')
    score_model = joblib.load('model-score.joblib')
    
    return ears_model, eyes_model, muzzle_model, whiskers_model, head_model, score_model

def classify_image (image_path, ears_model, eyes_model, muzzle_model, whiskers_model, head_model, score_model):
    try:
        print('\n\n', image_path)
        # load image
        image = cv2.imread(image_path)

        # crop the features
        print("Cropping the features…")
        ears, eyes, muzzle = keypoints.predict_and_crop(image_path)
        feature_scores = []

        # classify ears
        ears_preprocessed = preprocess_feature(ears, 128, 64)
        ears_score = score_feature (ears_preprocessed, ears_model)
        feature_scores.append(ears_score)

        # classify eyes
        eyes_preprocessed = preprocess_feature(eyes, 128, 64)
        eyes_score = score_feature (eyes_preprocessed, eyes_model)
        feature_scores.append(eyes_score)

        # classify muzzle
        muzzle_preprocessed = preprocess_feature(muzzle, 128, 64)
        muzzle_score = score_feature (muzzle_preprocessed, muzzle_model)
        feature_scores.append(muzzle_score)

        # classify whiskers
        whiskers_preprocessed = preprocess_feature(muzzle, 128, 64)  
        whiskers_score = score_feature (whiskers_preprocessed, whiskers_model)
        feature_scores.append(whiskers_score)

        # classify head
        head_preprocessed = preprocess_feature(image, 256, 256)
        head_score = score_feature (head_preprocessed, head_model)
        feature_scores.append(head_score)

        # calculate score
        feature_scores = np.array(feature_scores)
        feature_scores = feature_scores.reshape((1, 5))
        result = score_model.predict(feature_scores)
        result = np.round(result).astype(int)
        final_score = result[0]

        return feature_scores, final_score
    except:
        print("issue with image", image_path)
        
def main ():
    ears_model, eyes_model, muzzle_model, whiskers_model, head_model, score_model = load_models()
    
    # load labels
    csv_path = 'data-labels/labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
    labels.fillna(0, inplace=True)
    
    # set up dir path
    path = 'data-pain'
    dir = os.listdir(path)
    i = 0
    n = 20
    # iterate over images
    for image in dir:
        image_path = os.path.join(path, image)
        try:
            # get actual scores
            data = labels.loc[labels['imageid'] == image]
            data = np.array(data)
            data = data.squeeze()
            data = data[1:]
            print(data)
            
            # get predicted scores
            feature_scores, final_score = classify_image(image_path, ears_model, 
                                                         eyes_model, muzzle_model, 
                                                         whiskers_model, head_model, 
                                                         score_model)
            predicted = []
            predicted.append(feature_scores)
            predicted.append(final_score)
            predicted.squeeze()
            
            # print results
            print(image)
            print(data)
            print(predicted)
            print('\n')
        except:
            print("issue with image", image_path)
            
        if (i == n):
            return
        i+=1

main()
