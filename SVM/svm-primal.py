#!/usr/bin/env python3

import sys
import csv
import random

map_label = {'0': -1, '1': 1}

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
            labels.append(map_label[label])

    return attributes, labels

def svm_train(attributes, labels, C, T, gamma_0, a, schedule):
    num_features = len(attributes[0])
    weights = [0.0] * num_features
    bias = 0.0

    n = len(labels)
    t = 0  
    objective_values = []

    for epoch in range(T):
        combined = list(zip(attributes, labels))
        random.shuffle(combined)
        attributes_shuffled, labels_shuffled = zip(*combined)

        for i in range(n):
            x_i = attributes_shuffled[i]
            y_i = labels_shuffled[i]

            if schedule == 1:
                gamma_t = gamma_0 / (1 + (gamma_0 / a) * t)
            elif schedule == 2:
                gamma_t = gamma_0 / (1 + t)
            else:
                sys.exit(1)

            f_x = sum(w * x for w, x in zip(weights, x_i)) + bias

            if y_i * f_x >= 1:
                grad_w = [w_i for w_i in weights]
                grad_b = 0.0
            else:
                grad_w = [w_i - C * y_i * x_i_j for w_i, x_i_j in zip(weights, x_i)]
                grad_b = - C * y_i

            weights = [w_i - gamma_t * g_i for w_i, g_i in zip(weights, grad_w)]
            bias = bias - gamma_t * grad_b

            t += 1 

        obj = compute_obj(attributes, labels, weights, bias, C)
        objective_values.append(obj)

    return weights, bias, objective_values

def compute_obj(attributes, labels, weights, bias, C):
    n = len(labels)
    hinge_losses = []
    
    for x_i, y_i in zip(attributes, labels):
        linear_combination = sum(w_i * x_i_j for w_i, x_i_j in zip(weights, x_i))
        hinge_loss = max(0, 1 - y_i * (linear_combination + bias))
        hinge_losses.append(hinge_loss)
    
    norm_w_squared = sum(w_i ** 2 for w_i in weights)
    
    obj = 0.5 * norm_w_squared + C * sum(hinge_losses)

    return obj

def svm_predict(attributes, weights, bias):
    predictions = []
    for attribute in attributes:
        f_x = sum(w * x for w, x in zip(weights, attribute)) + bias
        prediction = 1 if f_x >= 0 else -1
        predictions.append(prediction)

    return predictions

def compute_error(labels, predictions):
    error_count = 0
    total = len(labels)

    for test_label, predicted_label in zip(labels, predictions):
        if test_label != predicted_label:
            error_count += 1

    error_rate = error_count / total

    return error_rate

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    T = int(sys.argv[3])
    C = float(sys.argv[4])
    gamma_0 = float(sys.argv[5])
    a = float(sys.argv[6])
    schedule = int(sys.argv[7])

    train_attributes, train_labels = load_data(train_data)
    test_attributes, test_labels = load_data(test_data)

    weights, bias, objective_values = svm_train(train_attributes, train_labels, C, T, gamma_0, a, schedule)

    train_predictions = svm_predict(train_attributes, weights, bias)
    train_error = compute_error(train_labels, train_predictions)

    test_predictions = svm_predict(test_attributes, weights, bias)
    test_error = compute_error(test_labels, test_predictions)

    print("Learned weights:", weights)
    print("Learned bias:", bias)
    print("Training error rate: {:.3f}".format(train_error))
    print("Test error rate: {:.3f} \n".format(test_error))

if __name__ == "__main__":
    main()
