#! /usr/bin/env python
import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt


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


def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=5).fit(X_train, y_train)
    # 5-fold
    score = cross_validation.cross_val_score(clf, X_test, y_test, cv=5)
    print score

if __name__ == '__main__':
    main()
