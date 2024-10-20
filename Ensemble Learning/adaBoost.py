#!/usr/bin/env python3

import sys
import csv
import math
import statistics
from collections import Counter, defaultdict

label_map = {'yes': 1, 'no': -1}
inverse_label_map = {1: 'yes', -1: 'no'}

class DecisionStump:
    def __init__(self):
        self.attribute = None
        self.polarity = 1
        self.threshold = None
        self.label = None
        self.branches = {}

    def train(self, attributes, labels, weights):
        num_features = len(attributes[0])
        min_error = float('inf')

        for feature_index in range(num_features):
            feature_values = [attribute[feature_index] for attribute in attributes]
            unique_values = set(feature_values)

            for value in unique_values:
                error = 0.0
                for i in range(len(attributes)):
                    prediction = None
                    if attributes[i][feature_index] == value:
                        prediction = 1
                    else:
                        prediction = -1

                    if prediction != labels[i]:
                        error += weights[i]

                if error > 0.5:
                    error = 1 - error
                    polarity = -1
                else:
                    polarity = 1

                if error < min_error:
                    min_error = error
                    self.polarity = polarity
                    self.attribute = feature_index
                    self.threshold = value

        self.branches = {}
        for label in set(labels):
            self.branches[label] = label

    def predict(self, attribute):
        attr_value = attribute[self.attribute]
        if self.polarity == 1:
            if attr_value == self.threshold:
                return 1
            else:
                return -1
        else:
            if attr_value != self.threshold:
                return 1
            else:
                return -1

def load_data(filename):
    attributes = []
    labels = []

    with open(filename, 'r') as f:
        data = csv.reader(f)
        for row in data:
            if not row:
                continue 

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
            original_value = attribute[index].strip().lower()
            if original_value == 'unknown':
                new_attribute[index] = 'unknown'
            else:
                try:
                    value = float(original_value)
                except ValueError:
                    new_attribute[index] = 'unknown'
                    continue

                median_value = medians[index]
                if value >= median_value:
                    new_attribute[index] = 'greater'
                else:
                    new_attribute[index] = 'less'

        new_attributes.append(new_attribute)

    return new_attributes


def initialize_weights(n_samples):
    return [1.0 / n_samples] * n_samples

def compute_error(y_true, y_pred, weights):
    error = 0.0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            error += weights[i]

    return error

def update_weights(weights, alpha, y_true, y_pred):
    new_weights = []
    for i in range(len(weights)):
        new_weight = weights[i] * math.exp(-alpha * y_true[i] * y_pred[i])
        new_weights.append(new_weight)

    total_weight = sum(new_weights)
    new_weights = [w / total_weight for w in new_weights]

    return new_weights

def get_error(y_true, y_pred):
    error_count = sum(1 for true_label, predicted_label in zip(y_true, y_pred) if true_label != predicted_label)
    total_labels = len(y_true)
    error_rate = error_count / total_labels

    return error_rate

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    max_iterations = int(sys.argv[3])

    train_attributes, train_labels = load_data(train_data)
    test_attributes, test_labels = load_data(test_data)

    try:
        train_labels_numeric = [label_map[label] for label in train_labels]
        test_labels_numeric = [label_map[label] for label in test_labels]
    except KeyError as e:
        print(f"Invalid label found: {e}. Labels should be 'yes' or 'no'.")
        sys.exit(1)

    numerical_indices = detect_numerical_attributes(train_attributes)

    if numerical_indices:
        medians = {}
        for index in numerical_indices:
            try:
                values = [float(attribute[index]) for attribute in train_attributes]
                median_value = statistics.median(values)
                medians[index] = median_value

            except ValueError:
                print(f"Non-numeric value found in numerical attribute index {index}.")
                sys.exit(1)

        train_attributes = binarize_attributes(train_attributes, numerical_indices, medians)
        test_attributes = binarize_attributes(test_attributes, numerical_indices, medians)

    n_samples = len(train_labels_numeric)
    weights = initialize_weights(n_samples)
    alphas = []
    stumps = []

    for t in range(1, max_iterations + 1):
        stump = DecisionStump()
        stump.train(train_attributes, train_labels_numeric, weights)
        train_predictions = [stump.predict(attr) for attr in train_attributes]

        error = compute_error(train_labels_numeric, train_predictions, weights)
        if error == 0:
            alpha = float('inf')  
        elif error >= 0.5:
            continue
        else:
            alpha = 0.5 * math.log((1 - error) / error)

        alphas.append(alpha)
        stumps.append(stump)

        weights = update_weights(weights, alpha, train_labels_numeric, train_predictions)

        train_pred_final = []
        for i in range(n_samples):
            total = 0
            for stump_i, alpha_i in zip(stumps, alphas):
                prediction = stump_i.predict(train_attributes[i])
                total += alpha_i * prediction
            final_prediction = 1 if total >= 0 else -1
            train_pred_final.append(final_prediction)

        train_error = get_error(train_labels_numeric, train_pred_final)

        test_pred_final = []
        for i in range(len(test_labels_numeric)):
            total = 0
            for stump_i, alpha_i in zip(stumps, alphas):
                prediction = stump_i.predict(test_attributes[i])
                total += alpha_i * prediction
            final_prediction = 1 if total >= 0 else -1
            test_pred_final.append(final_prediction)

        test_error = get_error(test_labels_numeric, test_pred_final)

        stump_train_error = get_error(train_labels_numeric, train_predictions)

        stump_test_predictions = [stump.predict(attr) for attr in test_attributes]
        stump_test_error = get_error(test_labels_numeric, stump_test_predictions)

        print(f"T: {t}: Train Error: {train_error:.3f}, Test Error: {test_error:.3f}, " 
              f"Stump Train Error: {stump_train_error:.3f}, Stump Test Error: {stump_test_error:.3f}")

if __name__ == "__main__":
    main()
