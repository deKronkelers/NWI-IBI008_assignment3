# author: Hendrik Werner s4549775
# author Constantin Blach s4329872

import xlrd
from pylab import *
from sklearn.metrics import roc_curve, accuracy_score

nr_of_data_points = 108

with xlrd.open_workbook(filename="./data/classprobs.xls") as book:
    # assignment 3.3.1
    sheet = book.sheet_by_index(0)
    data = empty((nr_of_data_points, 3))
    for row in range(nr_of_data_points):
        data[row] = sheet.row_values(rowx=row, start_colx=0)

    y = data[:, 0]
    X = data[:, 1:]


    # assignment 3.3.2
    def plot_roc_curve(y_true, y_score, title):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        f = plt.subplot(111)
        f.plot(fpr, tpr)
        f.set_title(title)
        f.set_xlabel("false positive rate")
        f.set_ylabel("true positive rate")
        show()


    plot_roc_curve(y, X[:, 0], "Classifier 1")
    plot_roc_curve(y, X[:, 1], "Classifier 2")

    # assignment 3.3.3
    def area_under_curve(y_score):
        sum = 0
        m = array(range(y.shape[0]))[y == 1]
        n = array(range(y.shape[0]))[y == 0]
        for i in m:
            for j in n:
                if y_score[i] > y_score[j]:
                    sum += 1
        return sum / (m.shape[0] * n.shape[0])

    print("AUC for classifier 1: {}".format(area_under_curve(X[:, 0])))
    print("AUC for classifier 1: {}".format(area_under_curve(X[:, 1])))

    # assignment 3.3.4
    def accuracy(y_score):
        predictions = [1 if score >= .5 else 0 for score in y_score]
        return accuracy_score(y, predictions)

    print("Accuracy of the first classifier: {}".format(accuracy(X[:, 0])))
    print("Accuracy of the second classifier: {}".format(accuracy(X[:, 1])))
