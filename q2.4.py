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


    # 5-fold
    for train_index, test_index in cross_validation.KFold(n=len(X), n_folds=5, shuffle=True):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        #print 'TRAIN:', train_index
        #print 'TEST:', test_index
        best_score = 0
        best_scores = None

        parameters = [50, 40, 30, 20, 10, 5, 4, 3, 2, 1]
        for mss in parameters:
            clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=mss)
            # 5-fold
            scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
            avg_score = sum(scores)/len(scores)
            if best_score < avg_score:
                best_score = avg_score
                best_scores = scores
                best_mss = mss
        print "Best: min_samples_split = %d\tscores = %s" % (best_mss, best_scores)


if __name__ == '__main__':
    main()
