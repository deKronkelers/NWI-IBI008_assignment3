# author: Hendrik Werner s4549775
# author Constantin Blach s4329872

import scipy.io

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
