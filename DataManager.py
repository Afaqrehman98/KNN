import csv

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self):
        self.labels = []
        self.features = []

    # This function reads data from two CSV files and stores the contents as lists of lists in the
    # instance variables "labels" and "features" using the "csv" module with a delimiter of ";".
    def import_data(self):
        self.labels = list(csv.reader(open('Images.csv'), delimiter=';'))
        self.features = list(csv.reader(open('EdgeHistogram.csv'), delimiter=';'))

    # This function processes data stored in the "labels" and "features" variables. It removes the
    # first row (header), extracts label and feature values using a list comprehension, and computes the count of
    # each unique label using pandas "value_counts" method.
    def process_data(self):
        self.labels = self.labels[1:]
        self.labels = [x[1] for x in self.labels]
        self.features = self.features[1:]
        self.features = [x[1:] for x in self.features]
        labels2 = pd.DataFrame(self.labels)
        Total = labels2.value_counts()

    # This function converts "labels" into integer codes with "LabelEncoder" and returns the
    # result in "labels_encoded". It maps original class labels to the integers in a dictionary.
    def generate_fitting_codes(self):
        le = preprocessing.LabelEncoder()
        self.labels_encoded = le.fit_transform(self.labels)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        return self.labels_encoded

    # This function  splits the "features" and "labels_encoded" data into training and testing sets using
    # the "train_test_split" function and stores the sets in "X_train_o", "X_test_o", "y_train_o", and "y_test_o"
    # instance variables respectively. The function returns the "features" instance variable.
    def split_data(self):
        self.X_train_o, self.X_test_o, self.y_train_o, self.y_test_o = train_test_split(self.features,
                                                                                        self.labels_encoded,
                                                                                        test_size=0.10, random_state=50)
        return self.features

    # The "list_to_float" function converts the "X_train_o", "X_test_o", "y_train_o", and "y_test_o" instance
    # variables from lists to numpy arrays with floating-point data type, using the "astype" method. The function
    # returns the converted instance variables as a tuple.
    def list_to_float(self):
        self.X_train_o = np.array(self.X_train_o).astype(np.float)
        self.X_test_o = np.array(self.X_test_o).astype(np.float)
        self.y_train_o = np.array(self.y_train_o).astype(np.float)
        self.y_test_o = np.array(self.y_test_o).astype(np.float)
        return self.X_train_o, self.y_train_o, self.X_test_o, self.y_test_o
