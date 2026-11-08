# Soft Computing: Neural Network Exercises

My neural network exercises from the Soft Computing course (HW2) at the University of Kashan.
Sina Mozaffari Rad, student ID 40021160214.

They start with the simplest single-neuron models and end with a small multi-layer network. Everything is
written in PyTorch: I define each model as an `nn.Module` and write the training loop myself (forward pass,
loss, backward pass, optimizer step), with my own stopping rule for convergence. The Iris dataset comes from
scikit-learn, which I also use for splitting, scaling and metrics.

## EX4: Loading and splitting the data → [`ex4_iris_data_split.py`](ex4_iris_data_split.py)

- Load the Iris dataset (150 samples, 4 features, 3 classes) and turn it into PyTorch tensors.
- Split it 80 / 20 into training and test sets with a fixed random seed.
- This is the data setup that the next exercises build on.

## EX5: Adaline for two classes → [`ex5_adaline_binary.py`](ex5_adaline_binary.py)

- **Data:** only Setosa and Versicolor, with labels mapped to +1 / −1 and standardised features.
- **Model:** Adaline, a single linear neuron trained on the mean squared error with SGD (learning rate 0.01).
- **Convergence:** training stops when the loss changes by less than 1e-5 between two iterations.
- **Experiments:** I train a separate model on each of the 6 pairs of features, so the decision boundary can
  be drawn in 2-D. The boundary is plotted every 50 iterations while it moves, and again on the training and
  test sets at the end, together with the error curve.

Setosa and Versicolor are linearly separable, so a single linear neuron is enough here. Some feature pairs
(for example petal length and petal width) separate the classes much more cleanly than others.

## EX6: Perceptron vs. Adaline on all three classes → [`ex6_perceptron_vs_adaline.py`](ex6_perceptron_vs_adaline.py)

- **Data:** all 4 features and all 3 classes, standardised.
- **Perceptron:** one linear layer with 3 outputs, trained with cross-entropy loss. It stops early as soon as
  no training sample is misclassified.
- **Adaline:** the same layer, trained on the MSE against one-hot targets, with the same loss-change stopping
  rule as in EX5.
- **Evaluation:** 5-fold cross-validation for both models, average test accuracy, and a confusion matrix.

Versicolor and Virginica overlap, so they are not linearly separable. That is where most of the mistakes in
the confusion matrix are, for both models.

## EX7: Multi-layer network for XOR → [`ex7_xor_mlp.py`](ex7_xor_mlp.py)

- **Problem:** XOR, the classic example that a single neuron cannot solve, because no straight line splits
  the two classes.
- **Model:** 2 → 4 → 4 → 1 with ReLU in the hidden layers and a sigmoid output.
- **Training:** binary cross-entropy and the Adam optimizer (learning rate 0.01) for 20,000 epochs, printing
  the loss every 1,000 epochs.
- **Output:** rounded predictions compared with the expected outputs, and plots of the real and predicted
  classes.

The hidden layers let the network bend the input space, so after training it gives the correct output for
all four inputs. This is the main difference from EX5 and EX6, where the models can only draw straight lines.

## Files

```
ex4_iris_data_split.py          Iris loading and train/test split
ex5_adaline_binary.py           Adaline, two classes, every pair of features
ex6_perceptron_vs_adaline.py    Perceptron and Adaline, three classes, 5-fold CV
ex7_xor_mlp.py                  Two-hidden-layer network for XOR
```

## Running

The Iris dataset is built into scikit-learn, so nothing has to be downloaded.

```bash
pip install torch scikit-learn matplotlib numpy
python ex7_xor_mlp.py
```
