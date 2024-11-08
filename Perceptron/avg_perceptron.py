#!/usr/bin/env python3

import sys
import csv

label_map = {'0': -1, '1': 1}

def load_data(filename):
    attributes = []
    labels = []

    with open(filename, 'r') as f:
        data = csv.reader(f)
        for row in data:
            if not row:
                continue

            attributes.append([float(value) for value in row[:-1]])

            label = row[-1].strip()
            labels.append(label_map[label])

    return attributes, labels

def average_perceptron_train(attributes, labels, max_epochs):
    num_features = len(attributes[0])  
    weights = [0.0] * num_features      
    bias = 0.0                          
    weights_sum = [0.0] * num_features  
    bias_sum = 0.0                      
    counter = 1                        

    for epoch in range(max_epochs):
        for i in range(len(labels)):
            activation = bias
            for j in range(num_features):
                activation += weights[j] * attributes[i][j]

            if labels[i] * activation <= 0:
                for j in range(num_features):
                    weights[j] += labels[i] * attributes[i][j]

                bias += labels[i]

            for j in range(num_features):
                weights_sum[j] += weights[j]

            bias_sum += bias
            counter += 1

    avg_weights = [w / counter for w in weights_sum]
    avg_bias = bias_sum / counter

    return avg_weights, avg_bias

def perceptron_predict(attributes, weights, bias):
    predictions = []
    for attribute in attributes:
        activation = bias
        for w, x in zip(weights, attribute):
            activation += w * x

        prediction = 1 if activation > 0 else -1
        predictions.append(prediction)

    return predictions

def compute_error(labels, predictions):
    error_count = 0
    for true_label, predicted_label in zip(labels, predictions):
        if true_label != predicted_label:
            error_count += 1

    error_rate = error_count / len(labels)

    return error_rate

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    max_epochs = int(sys.argv[3])

    train_attributes, train_labels = load_data(train_data)
    test_attributes, test_labels = load_data(test_data)

    avg_weights, avg_bias = average_perceptron_train(train_attributes, train_labels, max_epochs)

    print("Learned average weight vector:", avg_weights)

    predictions = perceptron_predict(test_attributes, avg_weights, avg_bias)
    error_rate = compute_error(test_labels, predictions)

    print(f"Average prediction error: {error_rate:.3f}")

if __name__ == "__main__":
    main()
