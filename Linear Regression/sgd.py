#!/usr/bin/env python3

import sys
import csv
import math
import random

def load_data(filename):
    attributes = []
    targets = []

    with open(filename, 'r') as f:
        data = csv.reader(f)
        for row in data:
            attributes.append([float(value) for value in row[:-1]])
            targets.append(float(row[-1]))

    return attributes, targets

def compute_cost(attributes, targets, weights):
    total_cost = 0.0
    n_samples = len(targets)

    for x, y in zip(attributes, targets):
        prediction = sum(w * xi for w, xi in zip(weights, x))
        error = prediction - y
        total_cost += error ** 2

    return total_cost / (2 * n_samples)

def compute_gradient(x, y, weights):
    prediction = sum(w * xi for w, xi in zip(weights, x))
    error = prediction - y
    gradient = [error * xi for xi in x]

    return gradient


def sgd(attributes, targets, learning_rate, max_iterations):
    n_features = len(attributes[0])
    weights = [0.0] * n_features
    cost_history = []
    n_samples = len(targets)

    for iteration in range(1, max_iterations + 1):
        idx = random.randint(0, n_samples - 1)
        x = attributes[idx]
        y = targets[idx]

        gradient = compute_gradient(x, y, weights)
        weights = [w - learning_rate * g for w, g in zip(weights, gradient)]

        cost = compute_cost(attributes, targets, weights)
        cost_history.append(cost)

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: Cost = {cost:.6f}")

        if iteration > 1 and abs(cost_history[-2] - cost_history[-1]) < 1e-6:
            print(f"Convergence achieved at iteration {iteration}.")
            break

    return weights, cost_history

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]

    train_attributes, train_targets = load_data(train_data)
    test_attributes, test_targets = load_data(test_data)

    train_attributes = standardize_features(train_attributes)
    test_attributes = standardize_features(test_attributes)

    learning_rate = 0.01
    max_iterations = 100000

    weights, cost_history = sgd(train_attributes, train_targets, learning_rate, max_iterations)

    print("\nLearned Weights:")
    for i, w in enumerate(weights):
        print(f"w[{i}] = {w:.6f}")

    print("\nCost vals at each step:")
    print("Iteration,Cost")
    for idx, cost in enumerate(cost_history):
        print(f"{idx+1},{cost:.6f}")

    final_train_cost = cost_history[-1]
    final_test_cost = compute_cost(test_attributes, test_targets, weights)
    print(f"\nFinal Cost on Training Data: {final_train_cost:.6f}")
    print(f"Final Cost on Test Data: {final_test_cost:.6f}")

def standardize_features(attributes):
    n_features = len(attributes[0])
    means = [0.0] * n_features
    std_devs = [0.0] * n_features
    n_samples = len(attributes)
    
    for attr in attributes:
        for i in range(n_features):
            means[i] += attr[i]
            
    means = [m / n_samples for m in means]
    
    for attr in attributes:
        for i in range(n_features):
            std_devs[i] += (attr[i] - means[i]) ** 2

    std_devs = [math.sqrt(sd / n_samples) if sd > 0 else 1.0 for sd in std_devs]
    
    standardized_attributes = []
    for attr in attributes:
        standardized_attr = []

        for i in range(n_features):
            standardized_value = (attr[i] - means[i]) / std_devs[i]
            standardized_attr.append(standardized_value)

        standardized_attributes.append(standardized_attr)
    
    return standardized_attributes

if __name__ == "__main__":
    main()
