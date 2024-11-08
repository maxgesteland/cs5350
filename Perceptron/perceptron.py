#!/usr/bin/env python3

import sys
import csv

map = {'0': -1, '1': 1}

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
            labels.append(map[label])

    return attributes, labels

def perceptron_train(attributes, labels, max_epochs):
    num_features = len(attributes[0]) 
    weights = [0.0] * num_features      
    bias = 0.0                     

    for epoch in range(max_epochs):
        for i in range(len(labels)):

            weighted_sum = 0.0
            for j in range(num_features):
                weighted_sum += weights[j] * attributes[i][j]

            activation = weighted_sum + bias

            if labels[i] * activation <= 0:
                for j in range(num_features):
                    weights[j] += labels[i] * attributes[i][j]

                bias += labels[i]

    return weights, bias

def perceptron_predict(attributes, weights, bias):
    predictions = []
    for attribute in attributes:
        activation = sum(w * x for w, x in zip(weights, attribute)) + bias
        prediction = 1 if activation > 0 else -1
        predictions.append(prediction)

    return predictions

def compute_error(labels, predictions):
    error_count = 0  
    total = len(labels)  

    for true_label, predicted_label in zip(labels, predictions):
        if true_label != predicted_label:
            error_count += 1  

    error_rate = error_count / total
    
    return error_rate

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    max_epochs = int(sys.argv[3])

    train_attributes, train_labels = load_data(train_data)
    test_attributes, test_labels = load_data(test_data)

    weights, bias = perceptron_train(train_attributes, train_labels, max_epochs)
    predictions = perceptron_predict(test_attributes, weights, bias)
    error_rate = compute_error(test_labels, predictions)

    print("Learned weights:", weights)
    print("Average prediction error: {:.3f}".format(error_rate))

if __name__ == "__main__":
    main()
