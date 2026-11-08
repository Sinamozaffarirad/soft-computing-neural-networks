import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Adaline(nn.Module):
    def __init__(self, input_dim, lr=0.01, threshold=1e-5):
        super(Adaline, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.lr = lr
        self.threshold = threshold
        self.converged = False

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, y, n_iter=2000, plot_interval=50, X_train=None, y_train=None, feature_names=None):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.errors = []

        for i in range(n_iter):
            outputs = self.forward(X)
            loss = criterion(outputs, y.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.errors.append(loss.item())

            if len(self.errors) > 1 and abs(self.errors[-2] - self.errors[-1]) < self.threshold:
                self.converged = True
                self.convergence_iter = i
                break

            if i % plot_interval == 0 and X_train is not None and y_train is not None:
                plot_decision_boundary(self, X_train, y_train, f'Decision Boundary at Iteration {i}', feature_names)

        return self

    def predict(self, X):
        with torch.no_grad():
            outputs = self.forward(X)
            return torch.where(outputs >= 0, 1, -1)

def plot_decision_boundary(model, X, y, title, feature_names):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01))
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    predictions = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=50, edgecolor='k')
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])

    if model.converged:
        plt.text(x_max - 1, y_min + 0.5, f'Converged at iteration {model.convergence_iter}', fontsize=12, color='red', ha='right')

    plt.show()

# Loading the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Using only two classes (0 and 1)
X = X[y != 2]
y = y[y != 2]
y = torch.where(torch.tensor(y) == 1, -1, 1).float()

# Standardizing features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std = torch.tensor(X_std, dtype=torch.float32)

# Create combinations of features
feature_combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
feature_names = iris.feature_names

for combo in feature_combinations:
    X_train, X_test, y_train, y_test = train_test_split(X_std[:, list(combo)], y, test_size=0.2, random_state=42)

    model = Adaline(input_dim=2, lr=0.01, threshold=1e-5).fit(X_train, y_train, n_iter=2000, plot_interval=50, X_train=X_train, y_train=y_train, feature_names=[feature_names[combo[0]], feature_names[combo[1]]])

    # Plot decision boundary for training data
    plot_decision_boundary(model, X_train, y_train, f'Training Decision Boundary for Features {feature_names[combo[0]]} & {feature_names[combo[1]]}',
                           [feature_names[combo[0]], feature_names[combo[1]]])

    # Plot decision boundary for testing data
    plot_decision_boundary(model, X_test, y_test, f'Testing Decision Boundary for Features {feature_names[combo[0]]} & {feature_names[combo[1]]}',
                           [feature_names[combo[0]], feature_names[combo[1]]])

# Plot the error graph
plt.plot(range(1, len(model.errors) + 1), model.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.title('Adaline - Learning rate 0.01')
plt.show()
