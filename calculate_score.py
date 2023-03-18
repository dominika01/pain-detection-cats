import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV


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

# perform linear regression and evaluate the results
def linear_regression(x_train, x_test, y_train, y_test):
    # train
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    
    # evaluate
    y_pred = lin_reg.predict(x_test)
    
    # round and convert predictions to integers
    y_pred = np.round(y_pred,0)
    y_pred = y_pred.astype(int)
    
    # calculate error
    lin_mse = mean_squared_error(y_test, y_pred)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_mse, lin_rmse)
    
    # cross validation score
    scores = cross_val_score(lin_reg, x_test, y_test, scoring="neg_mean_squared_error",cv=10)
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Standard deviation:',scores.std())
    
    # display first 10 values
    y_test_series = pd.Series(y_test[:10], name='y_test').reset_index(drop=True)
    y_pred_series = pd.Series(y_pred[:10], name='y_pred').reset_index(drop=True)
    df = pd.concat([y_test_series, y_pred_series], axis=1)
    print(df)

# perform logistic regression and evaluate the results
def logistic_regression(x_train, x_test, y_train, y_test):
    # train
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    
    # evaluate
    y_pred = log_reg.predict(x_test)
    
    # round and convert predictions to integers
    y_pred = np.round(y_pred,0)
    y_pred = y_pred.astype(int)
    
    # calculate error
    log_mse = mean_squared_error(y_test, y_pred)
    log_rmse = np.sqrt(log_mse)
    print(log_mse, log_rmse)
    
    # cross validation score
    scores = cross_val_score(log_reg, x_test, y_test, scoring="neg_mean_squared_error",cv=10)
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Standard deviation:',scores.std())
    
    # display first 10 values
    y_test_series = pd.Series(y_test[:10], name='y_test').reset_index(drop=True)
    y_pred_series = pd.Series(y_pred[:10], name='y_pred').reset_index(drop=True)
    df = pd.concat([y_test_series, y_pred_series], axis=1)
    print(df)

# perform decision tree regression and evaluate the results
def tree_regressor(x_train, x_test, y_train, y_test):
    # train
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(x_train, y_train)
    
    # evaluate
    y_pred = tree_reg.predict(x_test)
    
    # round and convert predictions to integers
    y_pred = np.round(y_pred,0)
    y_pred = y_pred.astype(int)
    
    # calculate error
    tree_reg_mse = mean_squared_error(y_test, y_pred)
    tree_reg_rmse = np.sqrt(tree_reg_mse)
    print(tree_reg_mse, tree_reg_rmse)
    
    # cross validation score
    scores = cross_val_score(tree_reg, x_test, y_test, scoring="neg_mean_squared_error",cv=10)
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Standard deviation:',scores.std())
    
    # display first 10 values
    y_test_series = pd.Series(y_test[:10], name='y_test').reset_index(drop=True)
    y_pred_series = pd.Series(y_pred[:10], name='y_pred').reset_index(drop=True)
    df = pd.concat([y_test_series, y_pred_series], axis=1)
    print(df)

# perform random forest regression and evaluate the results
def random_forest(x_train, x_test, y_train, y_test):
    # train
    for_reg = RandomForestRegressor()
    for_reg.fit(x_train, y_train)
    
    # evaluate
    y_pred = for_reg.predict(x_test)
    
    # round and convert predictions to integers
    y_pred = np.round(y_pred,0)
    y_pred = y_pred.astype(int)
    
    # calculate error
    for_reg_mse = mean_squared_error(y_test, y_pred)
    for_reg_rmse = np.sqrt(for_reg_mse)
    print(for_reg_mse, for_reg_rmse)
    
    # cross validation score
    scores = cross_val_score(for_reg, x_test, y_test, scoring="neg_mean_squared_error",cv=10)
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Standard deviation:',scores.std())
    
    # display first 10 values
    y_test_series = pd.Series(y_test[:10], name='y_test').reset_index(drop=True)
    y_pred_series = pd.Series(y_pred[:10], name='y_pred').reset_index(drop=True)
    df = pd.concat([y_test_series, y_pred_series], axis=1)
    print(df)

# perform grid search to tune random forest regression
def grid_search(x_train, x_test, y_train, y_test):
    # define hyperparameters to tune
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 6, 8]}

    # perform the search
    for_reg = RandomForestRegressor()
    grid_search = GridSearchCV(for_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)
    
    # select the best hyperparameters
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(best_params)
    
    # evaluate
    y_pred = best_model.predict(x_test)
    
    # round and convert predictions to integers
    y_pred = np.round(y_pred,0)
    y_pred = y_pred.astype(int)
    
    # calculate error
    for_reg_mse = mean_squared_error(y_test, y_pred)
    for_reg_rmse = np.sqrt(for_reg_mse)
    print(for_reg_mse, for_reg_rmse)
    
    # cross validation score
    scores = cross_val_score(for_reg, x_test, y_test, scoring="neg_mean_squared_error",cv=10)
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Standard deviation:',scores.std())
    
    # display first 10 values
    y_test_series = pd.Series(y_test[:10], name='y_test').reset_index(drop=True)
    y_pred_series = pd.Series(y_pred[:10], name='y_pred').reset_index(drop=True)
    df = pd.concat([y_test_series, y_pred_series], axis=1)
    print(df)
    
def main():
    x_train, x_test, y_train, y_test = get_data()
    print("Running the algorithm…")
    #linear_regression(x_train, x_test, y_train, y_test)
    #logistic_regression(x_train, x_test, y_train, y_test)
    #tree_regressor(x_train, x_test, y_train, y_test)
    #random_forest(x_train, x_test, y_train, y_test)
    grid_search(x_train, x_test, y_train, y_test)
    print("Done.\n")
    

main()