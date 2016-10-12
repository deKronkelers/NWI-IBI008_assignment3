# author: Hendrik Werner s4549775
# author Constantin Blach s4329872

import xlrd
from pylab import *
from sklearn.metrics import roc_curve

with xlrd.open_workbook(filename="./data/classprobs.xls") as book:
    # assignment 3.3.1
    sheet = book.sheet_by_index(0)
    data = empty((108, 3))
    for row in range(108):
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
