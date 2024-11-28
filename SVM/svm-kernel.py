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

def gaussian_kernel(X1, X2, gamma):
    sq_dists = np.sum(X1**2, axis=1).reshape(-1,1) + \
               np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    K = np.exp(-sq_dists / gamma)
    
    return K

def dual_objective(alpha, Q):
    obj = 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)

    return obj

def zero_constraint(alpha, y):

    return np.dot(alpha, y)

def svm_train_kernel(X, y, C, gamma):
    n_samples = X.shape[0]
    initial_alpha = np.zeros(n_samples)

    K = gaussian_kernel(X, X, gamma)
    Q = K * np.outer(y, y)

    bounds = [(0, C) for _ in range(n_samples)]

    constraints = {'type': 'eq', 'fun': zero_constraint, 'args': (y,)}

    res = minimize(fun=dual_objective,
                   x0=initial_alpha,
                   args=(Q,),
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-6})

  
    alpha = res.x

    sv = alpha > 1e-5

    alpha_sv = alpha[sv]
    y_sv = y[sv]
    K_sv = K[np.ix_(sv, sv)] 
    b = np.mean(y_sv - np.dot(K_sv, alpha_sv * y_sv))

    return alpha, b, sv, X[sv], y[sv]

def svm_predict_kernel(X_train_sv, y_train_sv, alpha_sv, b, X_test, gamma):
    K_test = gaussian_kernel(X_test, X_train_sv, gamma)
    decision = np.dot(K_test, alpha_sv * y_train_sv) + b
    predictions = np.where(decision >= 0, 1, -1)

    return predictions

def compute_error(labels, predictions):
    error_rate = np.mean(labels != predictions)

    return error_rate

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    C = float(sys.argv[3])
    gamma = float(sys.argv[4])

    X_train, y_train = load_data(train_data)
    X_test, y_test = load_data(test_data)

    alpha, b, sv, X_train_sv, y_train_sv = svm_train_kernel(X_train, y_train, C, gamma)
    alpha_sv = alpha[sv]

    train_predictions = svm_predict_kernel(X_train_sv, y_train_sv, alpha_sv, b, X_train, gamma)
    train_error = compute_error(y_train, train_predictions)

    test_predictions = svm_predict_kernel(X_train_sv, y_train_sv, alpha_sv, b, X_test, gamma)
    test_error = compute_error(y_test, test_predictions)

    print("Training error rate: {:.3f}".format(train_error))
    print("Test error rate: {:.3f} \n".format(test_error))

if __name__ == "__main__":
    main()
