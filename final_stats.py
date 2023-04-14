import cv2
import numpy as np
import tensorflow as tf
import keypoints
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
    if (confidence>0.99):
        score = 0
    #print("Score:", score)
    #print("Confidence:", confidence)
    #print(probability_array)
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
        # load image
        image = cv2.imread(image_path)

        # crop the features
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

        feature_scores = feature_scores.squeeze()
        result_array = []
        for score in feature_scores:
            result_array.append(score)
        result_array.append(final_score)
        return result_array
    except:
        print("issue with image", image_path)

def score(true, pred):
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(true, pred)))
    print('\nClassification Report\n')
    print(classification_report(true, pred, target_names=['Score 0', 'Score 1', 'Score 2']))
    
    # make a confusion matrix
    confusion = confusion_matrix(true, pred)
    print('Confusion Matrix\n')
    print(confusion)

    lables = ['0','1','2']    
    ax= plt.subplot()
    sns.heatmap(confusion, annot=True, fmt='g', ax=ax);

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(lables); ax.yaxis.set_ticklabels(lables);
    plt.show()

def main():
    ears_model, eyes_model, muzzle_model, whiskers_model, head_model, score_model = load_models()
    
    # load labels
    csv_path = 'data-labels/labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
    labels.fillna(0, inplace=True)
    
    # set up dir path
    path = 'data-pain'
    dir = os.listdir(path)
    
    # set up arrays
    ears_true = []
    ears_pred = []
    
    eyes_true = []
    eyes_pred = []
    
    muzzle_true = []
    muzzle_pred = []
    
    whiskers_true = []
    whiskers_pred = []
    
    head_true = []
    head_pred = []
    
    overall_true = []
    overall_pred = []
    
    i = 0
    # iterate over images
    for image in dir:
        image_path = os.path.join(path, image)
        try:
            # get actual scores
            data = labels.loc[labels['imageid'] == image]
            data = np.array(data)
            data = data.squeeze()
            data = data[1:]
            
            # get predicted scores
            predicted = classify_image(image_path, ears_model, eyes_model, muzzle_model, 
                                        whiskers_model, head_model, score_model)
            
            if predicted is None:
                predicted = [0, 0, 0, 0, 0, 0]
            
            # print results
            print('\n')
            print(i)
            print(image)
            print(data)
            print(predicted)
            print('\n')
            
            # save results
            ears_true.append(data[0])
            ears_pred.append(predicted[0])
            
            eyes_true.append(data[1])
            eyes_pred.append(predicted[1])
            
            muzzle_true.append(data[2])
            muzzle_pred.append(predicted[2])
            
            whiskers_true.append(data[3])
            whiskers_pred.append(predicted[3])
            
            head_true.append(data[4])
            head_pred.append(predicted[4])
            
            overall_true.append(data[5])
            overall_pred.append(predicted[5])
            
        except:
            print("issue with image", image_path)
        
        i += 1
    
    # score all   
    try:
        print("\nears")
        score(ears_true, ears_pred)
    except:
        print("ears issue")  
    
    try:
        print("\neyes")
        score(eyes_true, eyes_pred)
    except:
        print("eyes issue")
        
    try:
        print("\nmuzzle")
        score(muzzle_true, muzzle_pred)
    except:
        print("muzzle issue")
        
    try:
        print("\nwhiskers")
        score(whiskers_true, whiskers_pred)
    except:
        print("whiskers issue")
        
    try:
        print("\nhead")
        score(head_true, head_pred)
    except:
        print("head issue")
    
    try:
        print("\noverall")
        score(overall_true, overall_pred)
    except:
        print("overall issue")
        

main()