import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#   Loading the Dataset
iris = load_iris()
X, y = iris.data, iris.target
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

#   Splitting data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")
