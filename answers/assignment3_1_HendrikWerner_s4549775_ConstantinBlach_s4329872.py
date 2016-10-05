# author: Hendrik Werner s4549775
# author Constantin Blach s4329872

import scipy.io
from numpy import array
from sklearn import tree

# assignment 3.1.1
wine_data = scipy.io.loadmat("./data/wine.mat")

X = wine_data["X"]
attributes = [X[:, i] for i in range(X.shape[1])]
attribute_names = [nl[0] for nl in wine_data["attributeNames"][0]]
attribute_units = [
    "g/dm^3", "g/dm^3", "g/dm^3", "g/dm^3", "g/dm^3", "mg/dm^3", "mg/dm^3",
    "g/cm^3", "pH", "g/dm^3", "% vol."
]

classes = wine_data["y"]
class_names = [wine_data["classNames"][i, 0][0] for i in range(2)]

# assignment 3.1.2
decision_tree = tree.DecisionTreeClassifier(min_samples_split=100)
decision_tree = decision_tree.fit(X, classes)

tree.export_graphviz(decision_tree, out_file="tree.dot", class_names=class_names)

# assignment 3.1.3
test_wine = array([6.9 , 1.09, 0.06, 2.1, 0.0061, 12, 31, 0.99, 3.5, 0.64, 12])
print("The test wine is probably a {} wine.".format(("red", "white")[decision_tree.predict(test_wine.reshape(1, -1))[0]]))

# assignment 3.1.4
successful_classifications = 0
for i, wine in enumerate(X):
    if decision_tree.predict(wine.reshape(1, -1)) == classes[i]:
        successful_classifications += 1
print("The fitted decision tree classifies {}% of the learning data correctly.".format(
    successful_classifications * 100 / X.shape[0]
))
