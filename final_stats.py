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

# preprocessing a feature
def preprocess_feature (feature, x_size, y_size):
    resized = cv2.resize(feature, (x_size, y_size))
    feature_array = tf.keras.preprocessing.image.img_to_array(resized)
    reshaped = feature_array.reshape(1, x_size, y_size, 3)
    preprocessed = tf.keras.applications.vgg16.preprocess_input(np.array(reshaped))
    return preprocessed

# scoring a feature based on probability array
def score_feature (feature, model):
    probability_array = model.predict(feature)
    score = np.argmax(probability_array, axis=1)
    score = score[0]
    return score

# loading all models
def load_models ():
    ears_model = tf.keras.models.load_model('model-ears')
    eyes_model = tf.keras.models.load_model('model-eyes')
    muzzle_model = tf.keras.models.load_model('model-muzzle')
    whiskers_model = tf.keras.models.load_model('model-whiskers')
    head_model = tf.keras.models.load_model('model-head')
    score_model = joblib.load('model-score.joblib')
    
    return ears_model, eyes_model, muzzle_model, whiskers_model, head_model, score_model

# classifying all features of an image
def classify_image (image_path, ears_model, eyes_model, muzzle_model, whiskers_model, head_model, score_model):
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

    feature_scores = np.array(feature_scores)
    feature_scores = feature_scores.squeeze()
    result_array = []
    
    for score in feature_scores:
        result_array.append(score)
    
    if np.average(feature_scores) >= 1.2:
        result_array.append(2)
    else:
        result_array.append(final_score)
    return result_array

# generate a classification report
def score (true, pred):
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

# runs the entire code
def run_stats():
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
    class_0 = 0
    class_1 = 0
    class_2 = 0
    max_images = 100
    
    # iterate over images
    print('image loading')
    for img in dir:

        image_path = os.path.join(path, img)
        try:
            image = cv2.imread(image_path)
            
            if image is not None:
                # check image class
                image_class = labels.loc[labels['imageid'] == img, 'overall_impression']

                # use an equal number of images from each class
                if not image_class.empty:
                    image_class = image_class.iloc[0]    
                    if (image_class == 0.0 and class_0 < max_images):
                        class_0 += 1
                        if (i == max_images*3):
                            break
                        i+=1
                        
                        
                    elif (image_class == 1.0 and class_1 < max_images):
                        class_1 += 1
                        if (i == max_images*3):
                            break
                        i+=1
                        
                        
                    elif (image_class == 2.0 and class_2 < max_images):
                        class_2 += 1
                        if (i == max_images*3):
                            break
                        i+=1
                    
                    else:
                        continue
  
                    # get actual scores
                    data = labels.loc[labels['imageid'] == img]
                    data = np.array(data)
                    data = data.squeeze()
                    data = data[1:]
                    
                    # get predicted scores
                    predicted = classify_image(image_path, ears_model, eyes_model, muzzle_model, 
                                                whiskers_model, head_model, score_model)
                    
                    # error handling
                    if predicted is None:
                        predicted = [0, 0, 0, 0, 0, 0]
                    
                    # print results
                    print('\n')
                    print(i, img)
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
            print("error with", image_path)
        
    
    # score all   
    try:
        print("\noverall")
        score(overall_true, overall_pred)
    except:
        print("overall issue")
        
run_stats()

