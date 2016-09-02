#! /usr/bin/env python
import numpy as np
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
    print X
    print y
    print X.shape, y.shape

    idx = np.arange(X.shape[0])
    #np.random.seed(13)
    np.random.shuffle(idx)

    n_samples_train = int(X.shape[0] * 0.8)
    idx_train = idx[:n_samples_train]
    idx_test = idx[n_samples_train:]

    X_train = X[idx_train]
    y_train = y[idx_train]
    print 'train'
    print X_train
    print y_train
    print X_train.shape, y_train.shape

    X_test = X[idx_test]
    y_test = y[idx_test]
    print 'test'
    print X_test
    print y_test
    print X_test.shape, y_test.shape

    parameters = [50, 40, 30, 20, 10, 5, 4, 3, 2, 1]
    train_scores = []
    test_scores = []
    for mss in parameters:
        clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=mss).fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)

    plt.plot(parameters, train_scores, 'r-', label='train')
    plt.plot(parameters, test_scores, 'b-', label='test')
    plt.xlabel('min_samples_split')
    plt.ylabel('score')
    plt.suptitle("2.2")
    plt.legend()
    plt.grid(True)
    plt.savefig('q2.2.png')
    plt.show()


if __name__ == '__main__':
    main()
