# author: Hendrik Werner s4549775
# author Constantin Blach s4329872

import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import tree
from numpy import array

# assignment 3.2.1
wine_data = scipy.io.loadmat("./data/wine.mat")
X_train, X_test, classes_train, classes_test = train_test_split(
    wine_data["X"], wine_data["y"]
)

def classification_errors(X_train, y_train, X_test, y_test, depths):
    errors = []
    for depth in depths:
        decision_tree = tree.DecisionTreeClassifier(max_depth=depth)
        decision_tree.fit(X_train, y_train)
        errors.append(1 - decision_tree.score(X_test, y_test))
    return errors

depths = range(2, 21)
errors = classification_errors(X_train, classes_train, X_test, classes_test, depths)
f = plt.subplot(111)
f.plot(depths, errors)
f.set_xlabel("decision tree max depth")
f.set_ylabel("classification error")
plt.show()

# assignment 3.2.2
kf = KFold(n_splits=10, shuffle=True)
errors_list = []
for train_index, test_index in kf.split(wine_data["X"]):
    errors_list.append(classification_errors(
        wine_data["X"][train_index],
        wine_data["y"][train_index],
        wine_data["X"][test_index],
        wine_data["y"][test_index],
        depths
    ))

errors_list = array(errors_list)
average_errors = [col.mean() for col in (errors_list[:, i] for i in range(errors_list.shape[1]))]

f = plt.subplot(111)
f.plot(depths, average_errors)
f.set_xlabel("decision tree max depth")
f.set_ylabel("average classification error")
plt.show()
