# author: Hendrik Werner s4549775
# author Constantin Blach s4329872

import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree

# assignment 3.2.1
wine_data = scipy.io.loadmat("./data/wine.mat")
X_train, X_test, classes_train, classes_test = train_test_split(
    wine_data["X"], wine_data["y"]
)

depths = range(2, 21)
errors = []
for depth in depths:
    decision_tree = tree.DecisionTreeClassifier(max_depth=depth)
    decision_tree.fit(X_train, classes_train)
    errors.append(1 - decision_tree.score(X_test, classes_test))

f = plt.subplot(111)
f.plot(depths, errors)
f.set_xlabel("decision tree max depth")
f.set_ylabel("classification error")
plt.show()
