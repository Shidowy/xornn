import random

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1
epochs = 10000

def initialize_weights(input_size, hidden_size, output_size):
    W1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
    b1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)]]
    W2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
    b2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)]]
    return W1, b1, W2, b2

W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

def sigmoid(x):
    return 1 / (1 + (2.71828 ** -x))

def sigmoid_derivative(x):
    return x * (1 - x)

def dot_product(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

def add_bias(matrix, bias):
    return [[m + b for m, b in zip(row, bias[0])] for row in matrix]

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = add_bias(dot_product(X, W1), b1)
    A1 = [[sigmoid(z) for z in row] for row in Z1]
    Z2 = add_bias(dot_product(A1, W2), b2)
    A2 = [[sigmoid(z) for z in row] for row in Z2]
    return Z1, A1, Z2, A2

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def backward_propagation(X, y, A1, A2, W2):
    dA2 = [[a - target for a, target in zip(A2_row, y_row)] for A2_row, y_row in zip(A2, y)]
    dZ2 = [[da * sigmoid_derivative(a) for da, a in zip(dA2_row, A2_row)] for dA2_row, A2_row in zip(dA2, A2)]
    
    dW2 = dot_product(transpose(A1), dZ2)
    db2 = [[sum(row) for row in zip(*dZ2)]]

    dA1 = dot_product(dZ2, transpose(W2))
    dZ1 = [[da * sigmoid_derivative(a) for da, a in zip(dA1_row, A1_row)] for dA1_row, A1_row in zip(dA1, A1)]
    
    dW1 = dot_product(transpose(X), dZ1)
    db1 = [[sum(row) for row in zip(*dZ1)]]
    
    return dW1, db1, dW2, db2

for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
    
    loss = sum((target[0] - output[0]) ** 2 for target, output in zip(y, A2)) / len(y)
    
    dW1, db1, dW2, db2 = backward_propagation(X, y, A1, A2, W2)
    
    W1 = [[w - learning_rate * dw for w, dw in zip(W1_row, dW1_row)] for W1_row, dW1_row in zip(W1, dW1)]
    b1 = [[b - learning_rate * db for b, db in zip(b1[0], db1[0])]]
    W2 = [[w - learning_rate * dw for w, dw in zip(W2_row, dW2_row)] for W2_row, dW2_row in zip(W2, dW2)]
    b2 = [[b - learning_rate * db for b, db in zip(b2[0], db2[0])]]
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print("\nEvaluation:")
for i, sample in enumerate(X):
    _, _, _, A2 = forward_propagation([sample], W1, b1, W2, b2)
    print(f"Input: {sample}, Predicted: {[round(a[0]) for a in A2]}, Actual: {y[i]}")

