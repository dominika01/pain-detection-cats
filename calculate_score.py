import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
    # remove the 1st column with file names
    labels = labels.iloc[:, 1:]
    
    # replace NaN values with 0s
    labels.fillna(0, inplace=True)
    
    # one-hot encoding
    # One-hot encode the entire DataFrame
    labels = pd.get_dummies(labels)
    print("Done.")
    
    return labels

# split into train and test sets
def split_sets(labels):
    print("Splitting…")
    
    x_data = labels.drop(columns=['overall_impression'])
    y_data = labels['overall_impression']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data,
                                                                        test_size=0.3,
                                                                        random_state=42)
    print("Done.")
    return x_train, x_test, y_train, y_test

# load, preprocess, and split the data
def get_data():
    labels = load_labels()
    labels = preprocessing(labels)
    x_train, x_test, y_train, y_test = split_sets(labels)

    return x_train, x_test, y_train, y_test

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
    
    # display first 10 values
    y_test_series = pd.Series(y_test[:10], name='y_test').reset_index(drop=True)
    y_pred_series = pd.Series(y_pred[:10], name='y_pred').reset_index(drop=True)
    df = pd.concat([y_test_series, y_pred_series], axis=1)
    print(df)

def main():
    x_train, x_test, y_train, y_test = get_data()
    linear_regression(x_train, x_test, y_train, y_test)
    

main()