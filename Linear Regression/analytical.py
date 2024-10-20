#!/usr/bin/env python3

import sys
import csv

def main():
    file_path = sys.argv[1]

    features_matrix = []
    target_values = []

    with open(file_path, 'r') as f:
        data = csv.reader(f)
        for row in data:
            features_matrix.append([1.0] + [float(value) for value in row[:-1]])
            target_values.append(float(row[-1]))

    target_matrix = [[y] for y in target_values]

    features_transposed = []
    num_samples = len(features_matrix)
    num_features = len(features_matrix[0])

    for i in range(num_features):
        transposed_row = []

        for j in range(num_samples):
            transposed_row.append(features_matrix[j][i])

        features_transposed.append(transposed_row)

    features_product_matrix = []

    for i in range(num_features):
        result_row = []

        for j in range(num_features):
            sum_product = 0.0

            for k in range(num_samples):
                sum_product += features_transposed[i][k] * features_matrix[k][j]

            result_row.append(sum_product)

        features_product_matrix.append(result_row)

    n = len(features_product_matrix)
    inversion_matrix = [row[:] for row in features_product_matrix]
    identity_matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)  
            else:
                row.append(0.0)  
        identity_matrix.append(row)

    for fd in range(n):
        row_scaler = 1.0 / inversion_matrix[fd][fd]

        for j in range(n):
            inversion_matrix[fd][j] *= row_scaler
            identity_matrix[fd][j] *= row_scaler

    for i in range(0, fd):
        elimination_scaler = inversion_matrix[i][fd]
        
        for j in range(n):
            inversion_matrix[i][j] = inversion_matrix[i][j] - elimination_scaler * inversion_matrix[fd][j]
            identity_matrix[i][j] = identity_matrix[i][j] - elimination_scaler * identity_matrix[fd][j]

    for i in range(fd + 1, n):
        elimination_scaler = inversion_matrix[i][fd]
        
        for j in range(n):
            inversion_matrix[i][j] = inversion_matrix[i][j] - elimination_scaler * inversion_matrix[fd][j]
            identity_matrix[i][j] = identity_matrix[i][j] - elimination_scaler * identity_matrix[fd][j]

    inverse_xtx_matrix = identity_matrix

    features_target_product = []
    for i in range(num_features):
        result_row = []
        sum_product = 0.0
        
        for k in range(num_samples):
            sum_product += features_transposed[i][k] * target_matrix[k][0]

        result_row.append(sum_product)
        features_target_product.append(result_row)

    weights_result_matrix = []
    for i in range(len(inverse_xtx_matrix)):
        result_row = []
        sum_product = 0.0

        for j in range(len(features_target_product[0])):
            for k in range(len(features_target_product)):
                sum_product += inverse_xtx_matrix[i][k] * features_target_product[k][j]

            result_row.append(sum_product)

        weights_result_matrix.append(result_row)

    learned_weights = [w[0] for w in weights_result_matrix]

    print("Learned Weights:")
    for i, w in enumerate(learned_weights):
        print(f"w[{i}] = {w:.4f}")

if __name__ == "__main__":
    main()
