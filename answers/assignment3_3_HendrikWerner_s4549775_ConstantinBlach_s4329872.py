# author: Hendrik Werner s4549775
# author Constantin Blach s4329872

import xlrd
from pylab import *
from scipy.stats import binom
from sklearn.metrics import roc_curve, accuracy_score

with xlrd.open_workbook(filename="./data/classprobs.xls") as book:
    # assignment 3.3.1
    sheet = book.sheet_by_index(0)
    data = empty((sheet.nrows, 3))
    for row in range(sheet.nrows):
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
    predictions0 = [1 if score >= .5 else 0 for score in X[:, 0]]
    predictions1 = [1 if score >= .5 else 0 for score in X[:, 1]]
    print("Accuracy of the first classifier: {}".format(accuracy_score(y, predictions0)))
    print("Accuracy of the second classifier: {}".format(accuracy_score(y, predictions1)))


    # assignment 3.3.5
    def calc_confusion_matrix(predictions):
        confusion_matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        for i in range(y.shape[0]):
            true_class = y[i]
            predicted_class = predictions[i]
            if true_class == 1:
                if predicted_class == 1:
                    confusion_matrix["TP"] += 1
                else:
                    confusion_matrix["FP"] += 1
            else:
                if predicted_class == 0:
                    confusion_matrix["TN"] += 1
                else:
                    confusion_matrix["FN"] += 1
        return confusion_matrix


    print("Confusion matrix for classifier 1: {}".format(calc_confusion_matrix(predictions0)))
    print("Confusion matrix for classifier 2: {}".format(calc_confusion_matrix(predictions1)))

    s = 0  # classifier 1 > classifier 2
    f = 0  # classifier 2 > classifier 1

    for i in range(y.shape[0]):
        if predictions0[i] != predictions1[i]:
            if predictions0[i] == y[i]:
                s += 1
            else:
                f += 1

    print("Classifier 1 correctly predicted the class where classifier 2 failed in {} cases.".format(s))
    print("Classifier 2 correctly predicted the class where classifier 1 failed in {} cases.".format(f))

    print("p-value for classifier 2: {}".format(binom.cdf(min(s, f) + 1, s + f, .5)))
    print("p-value for classifier 1: {}".format(binom.sf(max(s, f) - 1, s + f, .5)))
