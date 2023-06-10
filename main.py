import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from DataManager import DataManager

dataManager = DataManager()

# The following lines import data, process it, and generate labels and feature
dataManager.import_data()
dataManager.process_data()
labels_encoded = dataManager.generate_fitting_codes()
features = dataManager.split_data()

# training and testing sets, and these sets are converted to floating point values.
X_train_o, y_train_o, X_test_o, y_test_o = dataManager.list_to_float()

#  The kf variable creates an instance of the KFold class with 5 splits, shuffle enabled, and a random state of 42.
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# The "setting_knn_range" function sets up a range of K-NN values, fits the K-NN model with each value, predicts the
# test data, calculates the accuracy and adds it to a list of accuracy scores.
def setting_knn_range():
    global K_nn_o, original, knn
    K_nn_o = np.arange(1, 31, 1)
    original = []
    K_nn_o = np.arange(1, 31, 1)
    original = []
    for vals in K_nn_o:
        knn = KNeighborsClassifier(n_neighbors=vals)
        knn.fit(X_train_o, y_train_o)
        y_pred_o = knn.predict(X_test_o)
        print(f"Accuracy at K_NN:{vals}", metrics.accuracy_score(y_test_o, y_pred_o))
        original.append(metrics.accuracy_score(y_test_o, y_pred_o))


# Setting K-NN range from 1 to 30
setting_knn_range()

# Concatenating and changing to dataframe
original_data = np.c_[K_nn_o, original]


# in this function the first column is renamed as "K-NN" and the second column is renamed as "Accuracy". The function
# returns the updated dataframe. The function is then called and passed the dataframe "original_data". The result is
# stored in "original_data".
def names(data):
    data = data.rename(columns={0: 'K-NN', 1: 'Accuracy'})
    return data


original_data = names(pd.DataFrame(original_data))


# The function creates a plot showing the relationship between K-NN values and accuracy in the original data. It
# includes labels, font size, a vertical line, and a title for the plot.
def visualize_accuracy():
    global fig, ax
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(original_data['K-NN'], original_data['Accuracy'], color='yellow', marker='o',
            markerfacecolor='red', markersize=12)
    ax.set_xlabel('K-NN Values', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    plt.axvline(x=8)
    ax.set_title('Accuracy for 101 Classes at different K-NN values', fontsize=20)


visualize_accuracy()

balancing_features = np.array(features)
balancing = np.c_[labels_encoded, balancing_features]

# Selecting number of images and k-nn values
No_Img = [3, 5, 10, 15]
K_nn = [1, 3, 5, 10, 15]


# The "empty_list_to_calculate_accuracies" function initializes global variables as empty lists and sets the "init"
# variable to False and "b_c" to the value of "balancing".
def empty_list_to_calculate_accuracies():
    global accuracy_3, accuracy_5, accuracy_10, accuracy_15, init, b_c
    accuracy_1 = []
    accuracy_3 = []
    accuracy_5 = []
    accuracy_10 = []
    accuracy_15 = []
    init = False
    b_c = balancing


empty_list_to_calculate_accuracies()


# The "looping_for_desired_values" function trains K-NN models with varying number of images and neighbors,
# balancing data based on labels. It loops and calculates accuracy scores, storing results in different lists based
# on number of images.
def looping_for_desired_values():
    global balancing, init, balancing_features, knn
    for values in No_Img:
        print('Images', values)
        for run in range(1, 6):
            balancing = b_c
            balanced = 0
            init = False

            for y in range(0, 101):
                y = str(y)
                index = -1
                count = 0
                for x in balancing[:, 0]:
                    index = index + 1
                    if y == x:
                        count = count + 1
                        if init == False:
                            balanced = balancing[index]
                            balancing = np.delete(balancing, index, axis=0)
                            init = True
                        else:
                            balanced = np.vstack((balanced, balancing[index]))
                            balancing = np.delete(balancing, index, axis=0)
                        if count == values:
                            break

            balanced_labels = balanced[:, 0]
            balanced_features = balanced[:, 1:]

            X_train = balanced[:, 1:]
            X_test = balancing[:, 1:]
            y_train = balanced[:, 0]
            y_test = balancing[:, 0]

            # Train and evaluate your model here

        balancing_labels = balancing[:, 0]
        balancing_features = balancing[:, 1:]

        X_train = np.array(X_train).astype(np.float)
        X_test = np.array(X_test).astype(np.float)
        y_train = np.array(y_train).astype(np.float)
        y_test = np.array(y_test).astype(np.float)

        for numbers in K_nn:
            knn = KNeighborsClassifier(n_neighbors=numbers)
            print('K-NN:', numbers)
            if values == 3:
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                print(f"Accuracy for Images:{values}", metrics.accuracy_score(y_test, y_pred))
                accuracy_3.append(metrics.accuracy_score(y_test, y_pred))

            elif values == 5:
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                print(f"Accuracy for Images:{values}", metrics.accuracy_score(y_test, y_pred))
                accuracy_5.append(metrics.accuracy_score(y_test, y_pred))

            elif values == 10:
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                print(f"Accuracy for Images:{values}", metrics.accuracy_score(y_test, y_pred))
                accuracy_10.append(metrics.accuracy_score(y_test, y_pred))

            elif values == 15:
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                print(f"Accuracy for Images:{values}", metrics.accuracy_score(y_test, y_pred))
                accuracy_15.append(metrics.accuracy_score(y_test, y_pred))


looping_for_desired_values()


# The function merges the accuracy scores of K-NN models into one data frame. It combines arrays of accuracy scores
# with "K_nn" values, adds a column indicating the number of images, and concatenates the data frames into one,
# "Images_all".

def merging_into_single_data_frame():
    global Images_all
    Img_3 = np.c_[K_nn, accuracy_3]
    Img_5 = np.c_[K_nn, accuracy_5]
    Img_10 = np.c_[K_nn, accuracy_10]
    Img_15 = np.c_[K_nn, accuracy_15]
    Img_3 = names(pd.DataFrame(Img_3))
    Img_3['Images'] = 3
    Img_5 = names(pd.DataFrame(Img_5))
    Img_5['Images'] = 5
    Img_10 = names(pd.DataFrame(Img_10))
    Img_10['Images'] = 10
    Img_15 = names(pd.DataFrame(Img_15))
    Img_15['Images'] = 15
    Images_all = pd.concat([Img_3, Img_5, Img_10, Img_15], ignore_index=True)


merging_into_single_data_frame()


def visualization():
    global fig, ax
    fig = plt.figure(figsize=(12, 6))
    k_img = ['3', '5', '10', '15']
    for i in range(4):
        ax = fig.add_subplot(1, 4, i + 1)
        plt.subplots_adjust(wspace=0.5)
        start = 5 * i
        end = 5 * (i + 1)
        ax.plot(Images_all['K-NN'][start:end], Images_all['Accuracy'][start:end], color='yellow', marker='o',
                markerfacecolor='red', markersize=12)
        ax.set_xlabel('K-NN Values', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim([0.290, 0.430])
        ax.set_xticks(np.arange(1, 16, 2))
        ax.tick_params(axis='both', labelsize=12)
        ax.set_title(f'Images {k_img[i]}', fontsize=12)


visualization()

plt.show()
