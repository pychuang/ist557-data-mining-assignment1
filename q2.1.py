#! /usr/bin/env python
import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus


def load_dataset():
    X = []
    y = []
    with open('wine.data') as f:
        for line in f:
            data = line.strip().split(',')
            label = int(data[0])
            sample = [float(x) for x in data[1:]]
            X.append(sample)
            y.append(label)
    return np.array(X), np.array(y)


def export_decision_tree(clf, png_file_name):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(png_file_name)


def build_decision_tree(X, y):
    clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=15).fit(X, y)
    export_decision_tree(clf, 'q2.1.png')


def main():
    X, y = load_dataset()
    print X
    print X.shape, y.shape
    build_decision_tree(X, y)


if __name__ == '__main__':
    main()
