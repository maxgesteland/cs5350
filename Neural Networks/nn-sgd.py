#!/usr/bin/env python3
import math
import sys
import random
import csv

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sig_div(x):
    return x * (1.0 - x)

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features = [float(x) for x in row[:-1]]
            label = float(row[-1])
            data.append((features, label))

    return data

class NN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size=1):
        self.W1 = [[random.gauss(0,1) for _ in range(input_size+1)] for _ in range(hidden1_size)]
        self.W2 = [[random.gauss(0,1) for _ in range(hidden1_size+1)] for _ in range(hidden2_size)]
        self.W3 = [[random.gauss(0,1) for _ in range(hidden2_size+1)] for _ in range(output_size)]

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

    def backward(self, x, y_star):
        x_input = [1.0] + x
        a1_input = [1.0] + self.a1_vals
        a2_input = [1.0] + self.a2_vals

        y = self.a3_vals[0]
        
        dL_dy = (y - y_star)
        dy_dz3 = sig_div(y)
        dL_dz3 = dL_dy * dy_dz3
        
        grad_W3 = []
        row_grad_3 = [dL_dz3 * val for val in a2_input]
        grad_W3.append(row_grad_3)
        
        dL_da2 = [dL_dz3 * self.W3[0][j+1] for j in range(len(self.a2_vals))]
        
        dL_dz2 = [dL_da2_j * sig_div(a) for dL_da2_j, a in zip(dL_da2, self.a2_vals)]
        
        grad_W2 = []
        for dL_dz2_j in dL_dz2:
            grad_W2.append([dL_dz2_j * val for val in a1_input])
        
        dL_da1 = [0.0] * len(self.a1_vals)
        for k, dL_dz2_k in enumerate(dL_dz2):
            for j in range(len(self.a1_vals)):
                dL_da1[j] += dL_dz2_k * self.W2[k][j+1]
        
        dL_dz1 = [dL_da1_j * sig_div(a) for dL_da1_j, a in zip(dL_da1, self.a1_vals)]
        
        grad_W1 = []
        for dL_dz1_j in dL_dz1:
            grad_W1.append([dL_dz1_j * val for val in x_input])
        
        return grad_W1, grad_W2, grad_W3

    def update_weights(self, grad_W1, grad_W2, grad_W3, eta):
        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                self.W1[i][j] -= eta * grad_W1[i][j]

        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                self.W2[i][j] -= eta * grad_W2[i][j]

        for i in range(len(self.W3)):
            for j in range(len(self.W3[i])):
                self.W3[i][j] -= eta * grad_W3[i][j]


def compute_error(nn, data):
    incorrect = 0
    for x, y in data:
        pred = nn.forward(x)
        pred_label = 1 if pred >= 0.5 else 0
        if pred_label != y:
            incorrect += 1

    return incorrect / len(data)


def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    hidden_size = int(sys.argv[3])
    gamma_0 = float(sys.argv[4])
    d = float(sys.argv[5])
    num_epochs = int(sys.argv[6])

    train_data = load_data(train_file)
    test_data = load_data(test_file)

    input_dim = len(train_data[0][0])
    output_dim = 1

    nn = NN(input_dim, hidden_size, hidden_size, output_dim)

    t = 0 
    for epoch in range(num_epochs):
        random.shuffle(train_data)
        for x, y in train_data:
            t += 1
            eta = gamma_0 / (1.0 + (gamma_0/d)*t)
            nn.forward(x)
            grad_W1, grad_W2, grad_W3 = nn.backward(x, y)
            nn.update_weights(grad_W1, grad_W2, grad_W3, eta)
        

    train_err = compute_error(nn, train_data)
    test_err = compute_error(nn, test_data)

    print("Final Test Error: {:.3f}".format(train_err))
    print("Final Test Error: {:.3f}".format(test_err))

if __name__ == "__main__":
    main()
