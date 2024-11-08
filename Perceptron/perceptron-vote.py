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

def voted_perceptron_train(attributes, labels, max_epochs):
    num_features = len(attributes[0])  
    weights_list = []                  
    counts = []                        
    weights = [0.0] * num_features   
    bias = 0.0                         
    c = 1                             

    for epoch in range(max_epochs):
        for i in range(len(labels)):
            activation = bias

            for j in range(num_features):
                activation += weights[j] * attributes[i][j]

            if labels[i] * activation <= 0:
                weights_list.append((weights.copy(), bias, c))

                for j in range(num_features):
                    weights[j] += labels[i] * attributes[i][j]

                bias += labels[i]
                c = 1
            else:
                c += 1

    weights_list.append((weights.copy(), bias, c))

    return weights_list

def voted_perceptron_predict(attributes, weights_list):
    predictions = []
    for attribute in attributes:
        vote_sum = 0.0

        for weights, bias, count in weights_list:
            activation = bias

            for w, x in zip(weights, attribute):
                activation += w * x
                
            prediction = 1 if activation > 0 else -1
            vote_sum += count * prediction

        final_prediction = 1 if vote_sum > 0 else -1
        predictions.append(final_prediction)

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

    weights_list = voted_perceptron_train(train_attributes, train_labels, max_epochs)

    print("List of distinct weight vectors with their counts:")
    for idx, (weights, bias, count) in enumerate(weights_list):
        print(f"Vector {idx + 1}:")
        print(f"  Weights: {weights}")
        print(f"  Count: {count}")

    predictions = voted_perceptron_predict(test_attributes, weights_list)
    error_rate = compute_error(test_labels, predictions)

    print(f"\nAverage prediction error: {error_rate:.3f}")

if __name__ == "__main__":
    main()
