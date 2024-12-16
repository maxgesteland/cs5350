#!/usr/bin/env python3
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    
    X_tens = torch.tensor(X, dtype=torch.float32)
    y_tens = torch.tensor(y, dtype=torch.float32)

    return X_tens, y_tens

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, depth, width, activation):
        super(MLP, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, width))
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        
        layers.append(nn.Linear(width, output_dim))
        
        self.layers = nn.ModuleList(layers)
        
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("Unknown activation: {}".format(activation))
        
        self.activation_type = activation
        
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers[:-1]: 
            if isinstance(layer, nn.Linear):
                if self.activation_type == "tanh":
                    nn.init.xavier_uniform_(layer.weight)

                elif self.activation_type == "relu":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

                nn.init.zeros_(layer.bias)
        
        if isinstance(self.layers[-1], nn.Linear):
            nn.init.xavier_uniform_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)

        x = self.layers[-1](x)

        return x

def train_model(model, optimizer, criterion, X_train, y_train, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        pred_proba = torch.sigmoid(pred)
        
        predicted_labels = (pred_proba >= 0.5).float()
        incorrect = (predicted_labels != y).float().sum().item()
        error_rate = incorrect / y.shape[0]

    return error_rate

def main():
    
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    activation = sys.argv[3]
    depth = int(sys.argv[4])
    width = int(sys.argv[5])
    epochs = int(sys.argv[6])
    

    X_train, y_train = load_data(train_data_path)
    X_test, y_test = load_data(test_data_path)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]  
    

    model = MLP(input_dim, output_dim, depth, width, activation)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    train_model(model, optimizer, criterion, X_train, y_train, epochs)
    
    train_error = evaluate_model(model, X_train, y_train)
    test_error = evaluate_model(model, X_test, y_test)
    
    print("Training classification error: {:.4f}".format(train_error))
    print("Test classification error: {:.4f}".format(test_error) +"\n")
    
if __name__ == "__main__":
    main()
