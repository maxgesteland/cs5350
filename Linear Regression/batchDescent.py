#!/usr/bin/env python3

import sys
import csv
import math

def load_data(filename):
    attributes = []
    targets = []

    with open(filename, 'r') as f:
        data = csv.reader(f)
        for row in data:
            attributes.append([float(value) for value in row[:-1]])
            targets.append(float(row[-1]))

    return attributes, targets

def initialize_weights(n_features):
    return [0.0] * n_features

def compute_cost(attributes, targets, weights):
    total_cost = 0.0
    n_samples = len(targets)

    for x, y in zip(attributes, targets):
        prediction = sum(w * xi for w, xi in zip(weights, x))
        error = prediction - y
        total_cost += error ** 2

    return total_cost / (2 * n_samples)

def compute_gradients(attributes, targets, weights):
    n_samples = len(targets)
    n_features = len(weights)
    gradients = [0.0] * n_features

    for i in range(n_features):
        gradient_sum = 0.0

        for x, y in zip(attributes, targets):
            prediction = sum(w * xi for w, xi in zip(weights, x))
            error = prediction - y
            gradient_sum += error * x[i]

        gradients[i] = gradient_sum / n_samples

    return gradients

def vector_norm(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def bgd(attributes, targets, learning_rate, tolerance):
    n_features = len(attributes[0])
    weights = initialize_weights(n_features)
    cost_history = []

    iteration = 0
    max_iterations = 100000  

    while iteration < max_iterations:
        gradients = compute_gradients(attributes, targets, weights)
        new_weights = [w - learning_rate * g for w, g in zip(weights, gradients)]
        norm_diff = vector_norm(new_weights, weights)
        weights = new_weights
        cost = compute_cost(attributes, targets, weights)
        cost_history.append(cost)

        if norm_diff < tolerance:
            break

        iteration += 1

    return weights, cost_history

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]

    train_attributes, train_targets = load_data(train_data)
    test_attributes, test_targets = load_data(test_data)

    learning_rates = [0.5]
    tolerance = 1e-5

    for r in learning_rates:
        weights, cost_history = bgd(train_attributes, train_targets, r, tolerance)
        
        print("Learned Weights:")
        for i, w in enumerate(weights):
            print(f"w[{i}] = {w:.6f}")

        print("\nCost function vals at each step:")
        print("Iteration,Cost")
        for idx, cost in enumerate(cost_history):
            print(f"{idx+1},{cost:.6f}")

        final_train_cost = cost_history[-1]
        final_test_cost = compute_cost(test_attributes, test_targets, weights)
        print(f"\nFinal Cost on Training Data: {final_train_cost:.6f}")
        print(f"Final Cost on Test Data: {final_test_cost:.6f}")

        break  


if __name__ == "__main__":
    main()
