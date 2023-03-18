import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


# load the labels into a dataframe
def load_labels():
    print("Loading labels…")
    # read data into a pandas dataframe
    csv_path = 'labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
    
    # replace NaN values with 0s
    labels.fillna(0, inplace=True)
    
    print("Done.\n")
    return labels

def split_sets (labels):
    x_data = labels.iloc[:, 1:-1]
    y_data = labels.iloc[:, -1:]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, 
                                                                        test_size=0.3, 
                                                                        random_state=42)
    
    return x_train, x_test, y_train, y_test

def graphs(labels):
    data = labels.iloc[:, 1:]

    attributes = ["ear_position","orbital_tightening","muzzle_tension",
                "whiskers_position", "head_position", "overall_impression"]
    
    # scatter_matrix(data,figsize=(12,8))
    corr_matrix=data.corr()
    corr_matrix["overall_impression"].sort_values(ascending=False)
    corr_matrix = corr_matrix.iloc[[-1],:]
    corr_matrix = corr_matrix.iloc[:, :-1]
    ax = corr_matrix.plot(kind='bar')
    for i in ax.containers:
        ax.bar_label(i, label_type='edge', labels=i.datavalues.round(2))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.show()
    

labels = load_labels()
graphs(labels)
