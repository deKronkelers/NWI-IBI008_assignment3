# author: Hendrik Werner s4549775
# author Constantin Blach s4329872

import matplotlib.pyplot as plt
import scipy.io
from numpy import array
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold


# Calculate classification errors for different tree depths.
def classification_errors(X_train, y_train, X_test, y_test, depths):
    errors_test = []
    errors_train = []
    for depth in depths:
        decision_tree = tree.DecisionTreeClassifier(max_depth=depth)
        decision_tree.fit(X_train, y_train)
        errors_test.append(1 - decision_tree.score(X_test, y_test))
        errors_train.append(1 - decision_tree.score(X_train, y_train))
    return errors_test, errors_train


# Plot the tree depth against the classification error.
def plot_errors(
        depths, errors_list,
        xlabel="decision tree max depth",
        ylabel="classification error"
):
    f = plt.subplot(111)
    for errors in errors_list:
        f.plot(depths, errors)
    f.set_xlabel(xlabel)
    f.set_ylabel(ylabel)
    plt.show()


# assignment 3.2.1
wine_data = scipy.io.loadmat("./data/wine.mat")
X = wine_data["X"]
y = wine_data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
depths = range(2, 21)
errors = classification_errors(X_train, y_train, X_test, y_test, depths)
plot_errors(depths, errors)

# assignment 3.2.2
kf = KFold(n_splits=10, shuffle=True)
errors_list_test = []
errors_list_train = []
for train_index, test_index in kf.split(X):
    errors_test, errors_train = classification_errors(
        X[train_index],
        y[train_index],
        X[test_index],
        y[test_index],
        depths
    )
    errors_list_test.append(errors_test)
    errors_list_train.append(errors_train)


def average_errors(errors_list):
    errors_list = array(errors_list)
    return [col.mean() for col in (errors_list[:, i] for i in range(errors_list.shape[1]))]


average_errors = (average_errors(errors_list_test), average_errors(errors_list_train))
plot_errors(depths, average_errors, ylabel="average classification error")
