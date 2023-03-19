import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


# load the labels into a dataframe
def load_labels():
    print("Loading labels…")
    
    # read data into a pandas dataframe
    csv_path = 'labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
        
    print("Done.\n")
    return labels

# check the correlation between input labels and output
def correlation(labels):
    data = labels.iloc[:, 1:]
    
    # create a correlation matrix
    corr_matrix  = data.corr()
    corr_matrix["overall_impression"].sort_values(ascending=False)
    
    # select the first column and last row only
    corr_matrix = corr_matrix.iloc[[-1],:]
    corr_matrix = corr_matrix.iloc[:, :-1]
    
    # create a plot
    ax = corr_matrix.plot(kind='bar')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i in ax.containers:
        ax.bar_label(i, label_type='edge', labels=i.datavalues.round(2))

    # show plot
    plt.show()
    
# preprocessing the data
def preprocessing(labels):
    print("Preprocessing…")
    
    # remove the 1st column with file names
    labels = labels.iloc[:, 1:]
    
    # replace NaN values with 0s
    labels.fillna(0, inplace=True)
    
    # one-hot encoding
    # One-hot encode the entire DataFrame
    labels = pd.get_dummies(labels)
    print("Done.\n")
    
    return labels

# split into train and test sets
def split_sets(labels):
    print("Splitting…")
    
    x_data = labels.drop(columns=['overall_impression'])
    y_data = labels['overall_impression']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                                        test_size=0.3,
                                                                        random_state=42)
    print("Done.\n")
    return x_train, x_test, y_train, y_test

# load, preprocess, and split the data
def get_data():
    labels = load_labels()
    labels = preprocessing(labels)
    x_train, x_test, y_train, y_test = split_sets(labels)

    return x_train, x_test, y_train, y_test

# display 10 actual and predicted results
def display(y_test, y_pred):
    y_test_series = pd.Series(y_test[:10], name='y_test').reset_index(drop=True)
    y_pred_series = pd.Series(y_pred[:10], name='y_pred').reset_index(drop=True)
    df = pd.concat([y_test_series, y_pred_series], axis=1)
    print(df)
    
def evaluate_model(model, x_test, y_test, y_pred):
    # round and convert predictions to integers
    y_pred = np.round(y_pred,0)
    y_pred = y_pred.astype(int)
    
    # calculate error
    lin_mse = mean_squared_error(y_test, y_pred)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_mse, lin_rmse)
    
    # cross validation score
    scores = cross_val_score(model, x_test, y_test, scoring="neg_mean_squared_error",cv=10)
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Standard deviation:',scores.std())
    
    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    #display(y_test, y_pred)
    
# perform linear regression and evaluate the results
def linear_regression(x_train, x_test, y_train, y_test):
    # train
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    
    # test
    y_pred = lin_reg.predict(x_test)
    evaluate_model(lin_reg, x_test, y_test, y_pred)

# perform logistic regression and evaluate the results
def logistic_regression(x_train, x_test, y_train, y_test):
    # train
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    
    # test
    y_pred = log_reg.predict(x_test)
    evaluate_model(log_reg, x_test, y_test, y_pred)

# perform decision tree regression and evaluate the results
def tree_regressor(x_train, x_test, y_train, y_test):
    # train
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(x_train, y_train)
    
    # test
    y_pred = tree_reg.predict(x_test)
    evaluate_model(tree_reg, x_test, y_test, y_pred)
    
# perform random forest regression and evaluate the results
def random_forest(x_train, x_test, y_train, y_test):
    # train
    for_reg = RandomForestRegressor()
    for_reg.fit(x_train, y_train)
    
    # test
    y_pred = for_reg.predict(x_test)
    evaluate_model(for_reg, x_test, y_test, y_pred)

# tune hyperparameters forrandom forest regression
def forest_random_search(x_train, x_test, y_train, y_test):
    # define hyperparameters to tune
    param_grid = {
        'n_estimators': randint(200, 300),
        'max_depth': randint(20, 50),
        'min_samples_split': randint(8, 12),
        'min_samples_leaf': randint(8, 12)}

    # perform the search
    for_reg = RandomForestRegressor()
    random_search = RandomizedSearchCV(for_reg, param_grid, n_iter=10, cv=5, random_state=42)
    random_search.fit(x_train, y_train)
    
    # select the best hyperparameters
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    print(best_params)
    
    # evaluate
    y_pred = best_model.predict(x_test)
    evaluate_model(best_model, x_test, y_test, y_pred)

# use support vector machines (SVMs) and evaluate the results
def svm(x_train, x_test, y_train, y_test):
    svm_model = SVC(kernel='linear', C=1, gamma='auto')
    svm_model.fit(x_train, y_train)
    
    # evaluate
    y_pred = svm_model.predict(x_test)
    evaluate_model(svm_model, x_test, y_test, y_pred)

# tune hyperparameters for SVMs
def svm_random_search(x_train, x_test, y_train, y_test):
    
    # define hyperparameters to tune
    param_grid = {'C': randint(1,100), 
                  'gamma': randint(1,100), 
                  'kernel': ['linear', 'rbf']}

    # train
    svm_model = SVC()
    grid_search = RandomizedSearchCV(svm_model, param_grid, n_iter=10, cv=5, random_state=42)
    grid_search.fit(x_train, y_train)
    
    # select the best hyperparameters
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(best_params)
    
    # evaluate
    y_pred = best_model.predict(x_test)
    evaluate_model(best_model, x_test, y_test, y_pred)

# create and evaluate a neural network    
def neural_network(x_train, x_test, y_train, y_test):
    # Define the model architecture
    model = Sequential()
    model.add(Dense(10, input_dim=5, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model to the training data
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test data
    scores = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    print(scores)

# create a neural network model
def create_model(neurons=1, optimizer='adam'):
    model = Sequential()
    model.add(Dense(2*neurons, input_dim=5, activation='relu'))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy', 'mse'])
    return model

# perform grid search on a neural network
def nn_grid_search(x_train, x_test, y_train, y_test):
    # Create the model
    model = KerasRegressor(build_fn=create_model, verbose=0)

    # Define the hyperparameters to search over
    params = {'neurons': [4, 8, 16, 32],
            'batch_size': [5, 16, 32, 50],
            'epochs': [10, 25, 50, 100],
            'optimizer': ['adam', 'sgd']}

    # Perform the grid search
    grid = GridSearchCV(estimator=model, param_grid=params, cv=3)
    grid_search = grid.fit(x_train, y_train)
    
    # select the best hyperparameters
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(best_params)
    
    # evaluate
    scores = best_model.evaluate(x_test, y_test)
    print(scores)

# run the code
def main():
    x_train, x_test, y_train, y_test = get_data()
    print("Running the algorithm…")
    
    #linear_regression(x_train, x_test, y_train, y_test)
    #logistic_regression(x_train, x_test, y_train, y_test)
    #tree_regressor(x_train, x_test, y_train, y_test)
    #random_forest(x_train, x_test, y_train, y_test)
    #forest_random_search(x_train, x_test, y_train, y_test)
    #svm(x_train, x_test, y_train, y_test)
    #svm_grid_search(x_train, x_test, y_train, y_test)
    #neural_network(x_train, x_test, y_train, y_test)
    nn_grid_search(x_train, x_test, y_train, y_test)
    
    print("Done.\n")

main()