#!/usr/bin/env python3

import math
import sys

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NN:
    def __init__(self, input_size, h1, h2):

        self.W1 = [
            [-1, -2, -3],
            [ 1,  2,  3]  
        ]
        

        self.W2 = [
            [-1, -2, -3], 
            [ 1,  2,  3]   
        ]
        

        self.W3 = [
            [-1, 2, -1.5] 
        ]

    def forward(self, x):
        x_input = [1.0] + x
        
        self.z1_vals = []
        self.a1_vals = []

        for w_row in self.W1:
            z = sum(w * xv for w, xv in zip(w_row, x_input))
            a = sigmoid(z)
            self.z1_vals.append(z)
            self.a1_vals.append(a)
        
        a1_input = [1.0] + self.a1_vals
        
        self.z2_vals = []
        self.a2_vals = []
        for w_row in self.W2:
            z = sum(w * av for w, av in zip(w_row, a1_input))
            a = sigmoid(z)
            self.z2_vals.append(z)
            self.a2_vals.append(a)
        
        a2_input = [1.0] + self.a2_vals
        
        self.z3_vals = []
        self.a3_vals = []
        for w_row in self.W3:
            z = sum(w * av for w, av in zip(w_row, a2_input))
            a = sigmoid(z)
            self.z3_vals.append(z)
            self.a3_vals.append(a)
        
        return self.a3_vals[0]

    def backward(self, x, label):
        x_input = [1.0] + x
        a1_input = [1.0] + self.a1_vals
        a2_input = [1.0] + self.a2_vals

        y = self.a3_vals[0]
        
        dL_dy = (y - label)
        dy_dz3 = sigmoid_derivative(y)
        dL_dz3 = dL_dy * dy_dz3
        
        grad_W3 = []
        row_grad_3 = [dL_dz3 * val for val in a2_input]
        grad_W3.append(row_grad_3)
        

        dL_da2 = [dL_dz3 * self.W3[0][j+1] for j in range(len(self.a2_vals))]
        
        dL_dz2 = [dL_da2_j * sigmoid_derivative(a) for dL_da2_j, a in zip(dL_da2, self.a2_vals)]
        
        grad_W2 = []
        for dL_dz2_j in dL_dz2:
            grad_W2.append([dL_dz2_j * val for val in a1_input])

        dL_da1 = [0.0]*len(self.a1_vals)
        for k, dL_dz2_k in enumerate(dL_dz2):
            for j in range(len(self.a1_vals)):
                dL_da1[j] += dL_dz2_k * self.W2[k][j+1]
        
        dL_dz1 = [dL_da1_j * sigmoid_derivative(a) for dL_da1_j, a in zip(dL_da1, self.a1_vals)]
        
        grad_W1 = []
        for dL_dz1_j in dL_dz1:
            grad_W1.append([dL_dz1_j * val for val in x_input])
        
        return grad_W1, grad_W2, grad_W3

def main():
    input_str = sys.argv[1]
    h1 = int(sys.argv[2])
    h2 = int(sys.argv[3])
    
    values = [float(v) for v in input_str.strip().split()]
    label = values[-1]
    x = values[:-1]

    input_size = len(x)
    nn = NN(input_size, h1, h2)
    
    y = nn.forward(x)
    grad_W1, grad_W2, grad_W3 = nn.backward(x, label)

    print("\nGradients for W3:")
    for k, row in enumerate(grad_W3):
        print(f"W3[{k}]: {row}")

    print("\nGradients for W2:")
    for j, row in enumerate(grad_W2):
        print(f"W2[{j}]: {row}")

    print("\nGradients for W1:")
    for i, row in enumerate(grad_W1):
        print(f"W1[{i}]: {row}")

if __name__ == "__main__":
    main()
