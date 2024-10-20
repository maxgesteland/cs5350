#!/usr/bin/env python3

import sys
import csv
from collections import Counter
import math
import statistics
import random

class Node:
    def __init__(self, is_leaf=False, label=None, attribute=None):
        self.is_leaf = is_leaf
        self.label = label
        self.attribute = attribute
        self.branches = {}
class DecisionTree:
    def __init__(self, split_method='entropy', max_depth=None, max_features=None):
        self.split_method = split_method
        self.max_depth = max_depth
        self.max_features = max_features  
        self.tree = None

    def train(self, attributes, labels):
        num_features = len(attributes[0])
        feature_indices = list(range(num_features))
        self.tree = self.id3(attributes, labels, feature_indices, depth=0)
            
    def id3(self, attributes, labels, feature_indices, depth):

        if len(set(labels)) == 1:
            return Node(is_leaf=True, label=labels[0])

        if (self.max_depth is not None and depth >= self.max_depth):
            most_common_label = Counter(labels).most_common(1)[0][0]
            return Node(is_leaf=True, label=most_common_label)

        if self.max_features and self.max_features < len(feature_indices):
            selected_features = random.sample(feature_indices, self.max_features)
        else:
            selected_features = feature_indices.copy()

        attr_to_split_on = self.choose_attr(attributes, labels, selected_features)
        if attr_to_split_on is None:
            most_common_label = Counter(labels).most_common(1)[0][0]
            return Node(is_leaf=True, label=most_common_label)

        new_root = Node(False, attribute=attr_to_split_on)
        attribute_values = [value[attr_to_split_on] for value in attributes]
        attribute_values_set = set(attribute_values)

        for value in attribute_values_set:
            attribute_subset, label_subset = self.get_subset(attributes, labels, attr_to_split_on, value)
            if attribute_subset:
                subtree = self.id3(attribute_subset, label_subset, feature_indices, depth + 1)
                new_root.branches[value] = subtree
            else:
                most_common_label = Counter(labels).most_common(1)[0][0]
                new_root.branches[value] = Node(is_leaf=True, label=most_common_label)

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
                if not label_subset:
                    continue
                weight = len(label_subset) / len(labels)
                purity = self.get_purity(label_subset)
                weighted_purity += weight * purity

            gain = node_purity - weighted_purity
            info_gains.append((gain, index))

        if not info_gains:
            return None

        max_gain, best_attr = max(info_gains, key=lambda x: x[0])
        
        if max_gain <= 0:
            return None
        else:
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
                    predictions.append(Counter(labels).most_common(1)[0][0] if labels else None)

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
    error_count = sum(1 for true_label, predicted_label in zip(y_true, y_pred) if true_label != predicted_label)
    total_labels = len(y_true)
    error_rate = error_count / total_labels

    return error_rate

def majority_vote(predictions_list):
    num_samples = len(predictions_list[0])
    aggregated_predictions = []

    for i in range(num_samples):
        votes = [predictions[i] for predictions in predictions_list]
        most_common = Counter(votes).most_common(1)[0][0]
        aggregated_predictions.append(most_common)

    return aggregated_predictions

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    num_trees = int(sys.argv[3])

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

    training_errors = []
    test_errors = []

    train_predictions_list = []
    test_predictions_list = []


    max_features_list = [2, 4, 6]

    for max_features in max_features_list:
        print(f"\nRunning Random Forest with max_features = {max_features}\n")
        training_errors = []
        test_errors = []

        train_predictions_list = []
        test_predictions_list = []

        for n in range(1, num_trees + 1):
            bootstrap_indices = [random.randint(0, len(train_attributes) - 1) for _ in range(len(train_attributes))]
            bootstrap_attributes = [train_attributes[i] for i in bootstrap_indices]
            bootstrap_labels = [train_labels[i] for i in bootstrap_indices]

            tree = DecisionTree(split_method='entropy', max_depth=None, max_features=max_features)
            tree.train(bootstrap_attributes, bootstrap_labels)

            train_predictions = tree.make_predictions(train_attributes)
            test_predictions = tree.make_predictions(test_attributes)

            train_predictions_list.append(train_predictions)
            test_predictions_list.append(test_predictions)

            aggregated_train_predictions = majority_vote(train_predictions_list)
            aggregated_test_predictions = majority_vote(test_predictions_list)

            train_error = get_error(train_labels, aggregated_train_predictions)
            test_error = get_error(test_labels, aggregated_test_predictions)

            training_errors.append(train_error)
            test_errors.append(test_error)

            print(f"Number of Trees: {n}, " + f"Train Error: {train_error:.3f}, " + f"Test Error: {test_error:.3f}")

if __name__ == "__main__":
    main()
