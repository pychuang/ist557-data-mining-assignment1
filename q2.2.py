#! /usr/bin/env python
import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt


def load_dataset():
    X = []
    Y = []
    with open('wine.data') as f:
        for line in f:
            data = line.strip().split(',')
            sample = [float(x) for x in data[:-1]]
            label = int(data[-1])
            X.append(sample)
            Y.append(label)
    return np.array(X), np.array(Y)


def main():
    X, Y = load_dataset()
    print X
    print Y
    print X.shape, Y.shape

    idx = np.arange(X.shape[0])
    #np.random.seed(13)
    np.random.shuffle(idx)

    n_samples_train = int(X.shape[0] * 0.8)
    idx_train = idx[:n_samples_train]
    idx_test = idx[n_samples_train:]

    X_train = X[idx_train]
    Y_train = Y[idx_train]
    print 'train'
    print X_train
    print Y_train
    print X_train.shape, Y_train.shape

    X_test = X[idx_test]
    Y_test = Y[idx_test]
    print 'test'
    print X_test
    print Y_test
    print X_test.shape, Y_test.shape

    parameters = [50, 20, 10, 5, 2, 1]
    train_scores = []
    test_scores = []
    for mss in parameters:
        clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=mss).fit(X_train, Y_train)
        train_score = clf.score(X_train, Y_train)
        test_score = clf.score(X_test, Y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)

    plt.plot(parameters, train_scores, 'r-')
    plt.plot(parameters, test_scores, 'b-')
    plt.xlabel('min_samples_split')
    plt.ylabel('score')
    plt.grid(True)
    plt.savefig('q2.2.png')
    plt.show()


if __name__ == '__main__':
    main()
