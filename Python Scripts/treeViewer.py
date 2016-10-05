from sklearn.externals.six import StringIO
from sklearn import tree
import pydot

FNAMES = ['Fixed acidity', 'Volatile acidity', 'Citric acid', 'Residual sugar', 'Chlorides', 'Free sulfur dioxide', 'Total sulfur dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']

def view(classifier):
    """ Renders a graph representation of classifier, and
        saves it to "MyTree.pdf" in the same folder
        as the executing script.
    """
    tree_dot = StringIO()
    tree.export_graphviz(classifier, out_file=tree_dot, feature_names=FNAMES)
    graph = pydot.graph_from_dot_data(tree_dot.getvalue())
    print "A"
    graph.write_pdf("MyTree.pdf")
    print "B"
