# author: Hendrik Werner s4549775
# author Constantin Blach s4329872

import scipy.io

# assignment 3.1.1
wine_data = scipy.io.loadmat("./data/wine.mat")
attribute_names = [nl[0] for nl in wine_data["attributeNames"][0]]

attribute_units = [
    "g/dm^3", "g/dm^3", "g/dm^3", "g/dm^3", "g/dm^3", "mg/dm^3", "mg/dm^3",
    "g/cm^3", "pH", "g/dm^3", "% vol."
]

# assignment 3.1.2
