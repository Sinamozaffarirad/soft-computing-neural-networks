import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Perceptron Model
class Perceptron(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.01, threshold=1e-5):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.lr = lr
        self.threshold = threshold
        self.converged = False

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, y, n_iter=200):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.errors = []

        for epoch in range(n_iter):
            outputs = self.forward(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.errors.append(loss.item())

            with torch.no_grad():
                predictions = torch.max(self.forward(X), 1)[1]
                misclassified = (predictions != y).sum().item()
                if misclassified == 0:
                    self.converged = True
                    self.convergence_epoch = epoch
                    break

        return self

    def predict(self, X):
        with torch.no_grad():
            outputs = self.forward(X)
            _, predicted = torch.max(outputs, 1)
            return predicted

# Loading the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardizing features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std = torch.tensor(X_std, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Performing k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
perceptron_accuracies = []

for train_index, test_index in kf.split(X_std):
    X_train, X_test = X_std[train_index], X_std[test_index]
    y_train, y_test = y[train_index], y[test_index]

    perceptron_model = Perceptron(input_dim=4, output_dim=3, lr=0.1)
    perceptron_model.fit(X_train, y_train, n_iter=500)

    y_pred_test = perceptron_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    perceptron_accuracies.append(test_accuracy)

print(f'Perceptron Model - Cross-Validation Testing Accuracies: {perceptron_accuracies}')
print(f'Perceptron Model - Average Testing Accuracy: {np.mean(perceptron_accuracies):.2f}')

cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.show()

# Adaline Model
class Adaline(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.01, threshold=1e-5):
        super(Adaline, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.lr = lr
        self.threshold = threshold
        self.converged = False

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, y, n_iter=200):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.errors = []

        y_onehot = torch.zeros(y.size(0), 3)
        y_onehot[torch.arange(y.size(0)), y] = 1

        for i in range(n_iter):
            outputs = self.forward(X)
            loss = criterion(outputs, y_onehot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.errors.append(loss.item())

            if len(self.errors) > 1 and abs(self.errors[-2] - self.errors[-1]) < self.threshold:
                self.converged = True
                self.convergence_epoch = i
                break

        return self

    def predict(self, X):
        with torch.no_grad():
            outputs = self.forward(X)
            _, predicted = torch.max(outputs, 1)
            return predicted

adaline_accuracies = []

for train_index, test_index in kf.split(X_std):
    X_train, X_test = X_std[train_index], X_std[test_index]
    y_train, y_test = y[train_index], y[test_index]

    adaline_model = Adaline(input_dim=4, output_dim=3, lr=0.01, threshold=1e-5)
    adaline_model.fit(X_train, y_train, n_iter=500)

    y_pred_test = adaline_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    adaline_accuracies.append(test_accuracy)

print(f'Adaline Model - Cross-Validation Testing Accuracies: {adaline_accuracies}')
print(f'Adaline Model - Average Testing Accuracy: {np.mean(adaline_accuracies):.2f}')

cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.show()