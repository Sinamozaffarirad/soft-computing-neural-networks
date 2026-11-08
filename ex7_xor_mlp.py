import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generating and prepare data
# Generating XOR data
x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Converting data to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Defining the neural network with an additional hidden layer
class XORNN(nn.Module):
    def __init__(self):
        super(XORNN, self).__init__()
        self.hidden1 = nn.Linear(2, 4)  # Input layer to first hidden layer
        self.hidden2 = nn.Linear(4, 4)  # First hidden layer to second hidden layer
        self.output = nn.Linear(4, 1)   # Second hidden layer to output layer
        self.activation = nn.ReLU()     # ReLU activation function

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Instantiating the model
model = XORNN()

# Specifying the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Using Adam optimizer with learning rate 0.01

# Training the model with updated training loop
num_epochs = 20000  # Number of epochs for training

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(x_tensor)  # Forward pass
    loss = criterion(outputs, y_tensor)  # Compute the loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update the weights

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the model
model.eval()
with torch.no_grad():
    predicted = model(x_tensor).numpy()
    predicted = np.round(predicted)  # Round predictions to 0 or 1

# Diagnostic prints
print("Test data:")
print(x)
print("Predictions:")
print(predicted)
print("Expected:")
print(y)

# Plotting the actual data
plt.figure(figsize=(10, 5))
# Plotting actual data points with distinct colors and adding them to the legend
plt.scatter(x[y[:, 0] == 0, 0], x[y[:, 0] == 0, 1], c='purple', label='Actual data 0', edgecolors='k', s=100, marker='o')
plt.scatter(x[y[:, 0] == 1, 0], x[y[:, 0] == 1, 1], c='yellow', label='Actual data 1', edgecolors='k', s=100, marker='o')
plt.legend(loc='best')
plt.title("XOR Problem - Actual Data")
plt.xlabel("Input Feature 1")
plt.ylabel("Input Feature 2")
plt.grid(True)
plt.show()

# Plotting the predicted data
plt.figure(figsize=(10, 5))
for i, point in enumerate(x):
    if predicted[i] == 1:
        plt.scatter(point[0], point[1], c='red', marker='P', s=200, edgecolors='k', label='Predicted data 1' if i == 0 else "")
    else:
        plt.scatter(point[0], point[1], c='blue', marker='P', s=200, edgecolors='k', label='Predicted data 0' if i == 0 else "")

# Adding legend entries manually 
plt.scatter([], [], c='red', marker='P', s=200, edgecolors='k', label='Predicted data 1')

plt.legend(loc='best')
plt.title("XOR Problem - Predicted Data")
plt.xlabel("Input Feature 1")
plt.ylabel("Input Feature 2")
plt.grid(True)
plt.show()
