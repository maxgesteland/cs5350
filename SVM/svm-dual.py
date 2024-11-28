#!/usr/bin/env python3

import sys
import csv
import numpy as np
from scipy.optimize import minimize

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

    return np.array(attributes), np.array(labels)

def dual_objective(alpha, X, y):
    K = np.dot(X, X.T)  
    y_matrix = np.outer(y, y)
    obj = 0.5 * np.dot(alpha, np.dot(K * y_matrix, alpha)) - np.sum(alpha)

    return obj

def zero_constraint(alpha, y):

    return np.dot(alpha, y)

def svm_train_dual(X, y, C):
    n_samples = X.shape[0]
    initial_alpha = np.zeros(n_samples)

    bounds = [(0, C) for _ in range(n_samples)]
    constraints = {'type': 'eq', 'fun': zero_constraint, 'args': (y,)}

    res = minimize(fun=dual_objective,
                   x0=initial_alpha,
                   args=(X, y),
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints)

    alpha = res.x

    w = np.sum((alpha * y)[:, np.newaxis] * X, axis=0)

    sv = (alpha > 1e-5) & (alpha < C - 1e-5)

    if np.any(sv):
        indices = np.where(sv)[0]
        b = np.mean(y[sv] - np.dot(X[sv], w))

    else:
        indices = np.where(alpha > 1e-5)[0]
        b = np.mean(y[indices] - np.dot(X[indices], w))

    return w, b, alpha

def svm_predict(X, w, b):
    activations = np.dot(X, w) + b
    predictions = np.where(activations >= 0, 1, -1)

    return predictions

def compute_error(labels, predictions):
    error_count = np.sum(labels != predictions)
    total = len(labels)
    error_rate = error_count / total

    return error_rate

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    C = float(sys.argv[3])

    train_attributes, train_labels = load_data(train_data)
    test_attributes, test_labels = load_data(test_data)

    w, b, alpha = svm_train_dual(train_attributes, train_labels, C)

    train_predictions = svm_predict(train_attributes, w, b)
    train_error = compute_error(train_labels, train_predictions)

    test_predictions = svm_predict(test_attributes, w, b)
    test_error = compute_error(test_labels, test_predictions)

    print("Learned weights:", w)
    print("Learned bias:", b)
    print("Training error rate: {:.3f}".format(train_error))
    print("Test error rate: {:.3f} \n".format(test_error))

if __name__ == "__main__":
    main()
