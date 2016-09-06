#! /usr/bin/env python
import numpy as np
from sklearn import cross_validation
from sklearn import tree
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
    print 'train:', X_train.shape, y_train.shape
    print y_train
    print 'test:', X_test.shape, y_test.shape
    print y_test

    parameters = xrange(1, 100)
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
