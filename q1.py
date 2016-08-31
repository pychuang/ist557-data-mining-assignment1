#! /usr/bin/env python
import math

class Node(object):

    def __init__(self, data):
        self.left = None
        self.right = None
        self.feature_to_split = None
        self.split_at = None

        # leaf node only
        self.data = data
        labels = [l for d, l in data]
        labels_count = sorted([(l, labels.count(l)) for l in set(labels)], key=lambda x: x[1], reverse=True)
        self.label = labels_count[0][0]


def entropy(data):
    labels = set(l for d, l in data)
    e = 0
    for label in labels:
        n = len([d for d, l in data if l == label])
        p = float(n) / len(data)
        if p == 0:
            continue
        e += p * math.log(1 / p, 2)
    return e


def split_node(node):
    print 'label', node.label, ':', node.data
    if len(node.data) < 15:
        return

    labels = set(l for d, l in node.data)
    if len(labels) == 1:
        return

    lowest_entropy = None

    features = len(node.data[0][0])
    for feature in xrange(features):
        values = sorted(set(d[feature] for d, l in node.data))
        for i in xrange(len(values) - 1):
            split_at = (values[i] + values[i+1]) / 2.0
            part1 = [(d, l) for d, l in node.data if d[feature] <= split_at]
            part2 = [(d, l) for d, l in node.data if d[feature] > split_at]

            e = entropy(part1) + entropy(part2)
            if not lowest_entropy or e < lowest_entropy:
                lowest_entropy = e
                node.left = Node(part1)
                node.right = Node(part2)
                node.feature_to_split = feature
                node.split_at = split_at

    print 'feature', node.feature_to_split, 'at', node.split_at
    node.data = None
    split_node(node.left)
    split_node(node.right)


def build_dt(dataset):
    root = Node(dataset)
    split_node(root)
    return root


def load_dataset():
    dataset = []
    with open('breast-cancer.data.txt') as f:
        for line in f:
            data = line.strip().split(',')
            data = [int(x) for x in data]
            data = (tuple(data[:-1]), data[-1])
            # ((feature1, feature2,...), label)
            dataset.append(data)
    return dataset


def main():
    dataset = load_dataset()
    #print dataset
    build_dt(dataset)


if __name__ == '__main__':
    main()
