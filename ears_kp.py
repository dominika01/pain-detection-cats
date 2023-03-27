
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from scipy.stats import randint
import keypoints

### UNDERSAMPLING APPROACH
# forest 36%
# svm 


# global variables
INPUT_SHAPE_X = 128
INPUT_SHAPE_Y = 64
ITERATION = '1'
EPOCHS = 10
BATCH_SIZE = 32

### DATA PREPROCESSING

# preprocess an individual image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray
    img = cv2.resize(img, (INPUT_SHAPE_X, INPUT_SHAPE_Y)) #resize
    img = img / 255.0 #normalize pixel values
    return img

# gets keypoint predictions and preprocesses them
def get_keypoints(model, image_path):
    kp = keypoints.predict(model, image_path)
    kp = kp.flatten()
    kp *= 255
    kp = kp.astype(int)
    kp = kp[:12]
    return kp
                
# load the labels into a dataframe
def load_labels():
    print("Loading labels…")
    # read data into a pandas dataframe
    csv_path = 'data-labels/labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
    
    # replace NaN values with 0s
    labels.fillna(0, inplace=True)
    
    print("Done.\n")
    return labels
    
# load the images and labels of ears  
def load_ears():
    # load labels
    labels = load_labels()
    
    print("Loading ears…")
    # set up the path
    ears_path = 'data/ears'
    ears_dir = os.listdir(ears_path)
    
    # init arrays
    x_ears = []
    y_ears = []
    
    # for undersampling purposes
    max_images = 1189
    class_0 = 0
    class_1 = 0
    class_2 = 0
    i = 0
    model = keypoints.load_model()
    # iterate over images
    for image in ears_dir:
        # get keypoints for an image
        image_path = os.path.join(ears_path, image)
        
        # check image class
        image_class = labels.loc[labels['imageid'] == image, 'ear_position']

        if not image_class.empty:
            image_class = image_class.iloc[0]    
            
            # append an equal number of images from each class
            if (image_class == 0.0 and class_0 < max_images):
                kp = get_keypoints(model, image_path)
                x_ears.append(kp)
                y_ears.append(image_class)
                class_0 += 1
                if (i == max_images*3):
                    break
                
            if (image_class == 1.0 and class_1 < max_images):
                kp = get_keypoints(model, image_path)
                x_ears.append(kp)
                y_ears.append(image_class)
                class_1 += 1
                if (i == max_images*3):
                    break
                
            if (image_class == 2.0 and class_2 < max_images):
                kp = get_keypoints(model, image_path)
                x_ears.append(kp)
                y_ears.append(image_class)
                class_2 += 1
                if (i == max_images*3):
                    break
            i+=1           
        
    x_ears = np.array(x_ears)
    y_ears = np.array(y_ears)
    
    x_ears = x_ears.reshape(max_images*3,12)
    y_ears = y_ears.astype(int)

    print("Done.\n")
    return x_ears, y_ears
    
# split the data into train, validation and evaluation sets
def split_data(x_data, y_data):
    print("Splitting data…")
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, 
                                                        y_data,
                                                        test_size=0.2,
                                                        random_state=42)
    
    print("Done.\n")
    return x_train, y_train, x_test, y_test
        
### BUILING & EVALUATING THE MODEL
# perform random forest regression and evaluate the results
def random_forest(x_train, x_test, y_train, y_test):
    # train
    for_reg = RandomForestRegressor()
    for_reg.fit(x_train, y_train)
    
    # test
    confusion_matrix(for_reg, x_test, y_test)

# tune hyperparameters forrandom forest regression
def forest_random_search(x_train, x_test, y_train, y_test):
    # define hyperparameters to tune
    param_grid = {
        'n_estimators': randint(10, 300),
        'max_depth': randint(10, 100),
        'min_samples_split': randint(5, 20),
        'min_samples_leaf': randint(5, 20)}

    # perform the search
    for_reg = RandomForestRegressor()
    random_search = RandomizedSearchCV(for_reg, param_grid, n_iter=20, cv=5, random_state=42)
    random_search.fit(x_train, y_train)
    
    # select the best hyperparameters
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    print(best_params)
    
    # evaluate
    confusion_matrix(best_model, x_test, y_test)
    
# tune hyperparameters for SVMs
def svm_random_search(x_train, x_test, y_train, y_test):
    
    # define hyperparameters to tune
    param_grid = {'C': randint(50,100), 
                  'gamma': randint(1,10), 
                  'kernel': ['linear', 'rbf']}

    # train
    svm_model = SVC()
    #grid_search = RandomizedSearchCV(svm_model, param_grid, n_iter=10, cv=5, random_state=42)
    print("searching…")
    svm_model.fit(x_train, y_train)
    
    # select the best hyperparameters
    #best_params = grid_search.best_params_
    #best_model = grid_search.best_estimator_
    #print(best_params)
    
    # evaluate
    confusion_matrix(svm_model, x_test, y_test)

def confusion_matrix(model, x_test, y_test):
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred,0)
    y_pred = y_pred.astype(int)
    y_pred.squeeze()
    print(y_test, y_pred)
    
    confusion = confusion_matrix(y_test,y_pred)
    print('Confusion Matrix Tree\n')
    print(confusion)

    #importing accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['Score 0', 'Score 1', 'Score 2']))
    
    import seaborn as sns

    lables = ['0','1','2']    

    ax= plt.subplot()

    sns.heatmap(confusion, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(lables); ax.yaxis.set_ticklabels(lables);
    plt.show()

# perform decision tree regression and evaluate the results
def tree_regressor(x_train, x_test, y_train, y_test):
    # train
    
    parameters={"splitter":["best","random"],
                "max_depth" : randint(1,10),
                "min_samples_leaf":randint(1,10),
                "min_weight_fraction_leaf": [0.1,0.3,0.5,0.7,0.9],
                "max_features":["log2","sqrt",None],
                "max_leaf_nodes": randint(1,100) }
    tree_reg = DecisionTreeRegressor()
    tuning_model=RandomizedSearchCV(tree_reg,parameters, n_iter=10, cv=5, random_state=42)
    tuning_model.fit(x_train, y_train)
    
    # select the best hyperparameters
    best_params = tuning_model.best_params_
    best_model = tuning_model.best_estimator_
    print(best_params)
    
    # test
    
    tree_reg.fit(x_train, y_train)
    confusion_matrix(tree_reg, x_test, y_test)
  
# create and evaluate a neural network    
def neural_network(x_train, x_test, y_train, y_test):
    # Define the model architecture
    model = Sequential()
    model.add(Dense(10, input_dim=12, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
                  metrics=['accuracy'])
    weights = {0: 2, 1: 1, 2: 2}
    # Fit the model to the training data
    model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2, class_weight=weights)

    # Evaluate the model on the test data
    scores = model.evaluate(x_test, y_test)
    
    print(scores)
    confusion_matrix(model, x_test, y_test)


def ears():
    # get the data
    x_ears, y_ears = load_ears()
    
    x_train, y_train, x_test, y_test = split_data(x_ears, y_ears)
    #forest_random_search(x_train, x_test, y_train, y_test)
    #svm_random_search(x_train, x_test, y_train, y_test)
    #tree_regressor(x_train, x_test, y_train, y_test)

    
    
    
### MAIN
def main():
    ears()
    

main()
