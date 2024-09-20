#!/usr/bin/env python3

import sys
import csv
from collections import Counter
import math
import statistics

class Node:
    def __init__(self, is_leaf=False, label=None, attribute=None):
        self.is_leaf = is_leaf
        self.label = label
        self.attribute = attribute
        self.branches = {}

class DecisionTree:
    def __init__(self, split_method='entropy', max_depth=None):
        self.split_method = split_method
        self.max_depth = max_depth
        self.tree = None

    def train(self, attributes, labels):
        num_features = len(attributes[0])
        feature_indices = list(range(num_features))

        self.tree = self.id3(attributes, labels, feature_indices, depth=0)

    def id3(self, attributes, labels, feature_indices, depth):
        if len(set(labels)) == 1:
            is_leaf = True
            label = labels[0]

            return Node(is_leaf, label)

        no_attributes_left = (attributes is None or len(attributes[0]) == 0)

        max_depth_reached = (self.max_depth is not None and depth >= self.max_depth)

        if no_attributes_left or max_depth_reached:
            most_common_label = Counter(labels).most_common(1)[0][0]

            return Node(is_leaf=True, label=most_common_label)

        attr_to_split_on = self.choose_attr(attributes, labels, feature_indices)

        new_root = Node(False, attribute=attr_to_split_on)

        attribute_values = []

        for value in attributes:
            attribute_value = value[attr_to_split_on]
            attribute_values.append(attribute_value)

        attribute_values_set = set(attribute_values)
        
        for value in attribute_values_set:
            attribute_subset, label_subset = self.get_subset(attributes, labels, attr_to_split_on, value)
            if attribute_subset:
                remaining_attributes = feature_indices.copy()
                remaining_attributes.remove(attr_to_split_on)
                subtree = self.id3(attribute_subset, label_subset, remaining_attributes, depth + 1)
                new_root.branches[value] = subtree
            else:
                most_common_label = Counter(labels).most_common(1)[0][0]
                new_root.branches[value] = Node(True, most_common_label)

        return new_root

    def get_subset(self, attributes, labels, index, value):
        attribute_subset = []
        label_subset = []

        for attribute, label in zip(attributes, labels):
            if attribute[index] == value:
                attribute_subset.append(attribute)
                label_subset.append(label)

        return attribute_subset, label_subset

    def choose_attr(self, attributes, labels, feature_indices):
        node_purity = self.get_purity(labels)

        info_gains = []
        for index in feature_indices:
            possible_values = [attribute[index] for attribute in attributes]
            attr_values = set(possible_values)

            weighted_purity = 0.0
            for value in attr_values:
                x, label_subset = self.get_subset(attributes, labels, index, value)

                if len(label_subset) == 0:
                    continue

                weight = len(label_subset) / len(labels)
                purity = self.get_purity(label_subset)
                weighted_purity += weight * purity

            gain = node_purity - weighted_purity
            info_gains.append((gain, index))

        max_gain = float('-inf')
        best_attr = None

        for gain, attr in info_gains:
            if gain > max_gain:
                max_gain = gain
                best_attr = attr

        return best_attr

    def get_purity(self, labels):
        counts = Counter(labels)
        total = len(labels)

        if self.split_method == 'entropy':
            entropy = 0.0
            for count in counts.values():
                p = count / total
                entropy -= p * math.log2(p)

            return entropy

        elif self.split_method == 'gini':
            gini = 1.0
            for count in counts.values():
                p = count / total
                gini -= p ** 2

            return gini

        elif self.split_method == 'majority_error':
            most_common = counts.most_common(1)[0][1]
            error = 1 - (most_common / total)

            return error

    def make_predictions(self, attributes):
        predictions = []

        for attribute in attributes:
            node = self.tree

            while not node.is_leaf:
                attr_value = attribute[node.attribute]

                if attr_value in node.branches:
                    node = node.branches[attr_value]
                else:
                    labels = self.collect_labels(node)
                    if labels:
                        predictions.append(Counter(labels).most_common(1)[0][0])
                    else:
                        predictions.append(None)
                    break

            if node.is_leaf:
                predictions.append(node.label)

        return predictions

    def collect_labels(self, node):
        if node.is_leaf:

            return [node.label]
        else:
            labels = []
            for child in node.branches.values():
                labels.extend(self.collect_labels(child))

            return labels

def load_data(filename):
    attributes = []
    labels = []

    with open(filename, 'r') as f:
        data = csv.reader(f)
        for row in data:
            attributes.append(row[:-1])
            labels.append(row[-1])

    return attributes, labels

def detect_numerical_attributes(attributes):
    columns = list(zip(*attributes))
    numerical_indices = []
    for i, column in enumerate(columns):
        is_numerical = True
        for value in column:
            try:
                float(value)
            except ValueError:
                is_numerical = False
                break
        if is_numerical:
            numerical_indices.append(i)

    return numerical_indices

def binarize_attributes(attributes, numerical_indices, medians):
    new_attributes = []
    for attribute in attributes:
        new_attribute = attribute.copy()
        for index in numerical_indices:
            value = float(new_attribute[index])
            median_value = medians[index]
            if value >= median_value:
                new_attribute[index] = 'greater'
            else:
                new_attribute[index] = 'less'
        new_attributes.append(new_attribute)

    return new_attributes

def get_error(y_true, y_pred):
    error_count = 0

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label != predicted_label:
            error_count += 1

    total_labels = len(y_true)

    error_rate = error_count / total_labels

    return error_rate

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    split_method = sys.argv[3] 
    max_depth = int(sys.argv[4])

    train_attributes, train_labels = load_data(train_data)
    test_attributes, test_labels = load_data(test_data)

    numerical_indices = detect_numerical_attributes(train_attributes)

    if numerical_indices:
        medians = {}
        for index in numerical_indices:
            values = [float(attribute[index]) for attribute in train_attributes]
            median_value = statistics.median(values)
            medians[index] = median_value

        train_attributes = binarize_attributes(train_attributes, numerical_indices, medians)
        test_attributes = binarize_attributes(test_attributes, numerical_indices, medians)

    tree = DecisionTree(split_method, max_depth)

    tree.train(train_attributes, train_labels)

    training_predictions = tree.make_predictions(train_attributes)
    test_predictions = tree.make_predictions(test_attributes)

    train_error = get_error(train_labels, training_predictions)
    test_error = get_error(test_labels, test_predictions)

    print(f"Train Error: {train_error:.3f}")
    print(f"Test Error: {test_error:.3f}")

if __name__ == "__main__":
    main()
