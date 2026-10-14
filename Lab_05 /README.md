# ANN Backpropagation from Scratch (XOR Problem)

A simple implementation of a feed-forward Artificial Neural Network (ANN) trained with the **backpropagation** algorithm, built from scratch using only **NumPy**. The network learns to solve the classic **XOR problem**, which a single perceptron cannot solve because the data is not linearly separable.

## Overview

- **Notebook:** `ANN_Back_Propagation.ipynb`
- **Language:** Python 3
- **Library:** NumPy
- **Task:** Learn the XOR logic gate
- **Environment:** Google Colab / Jupyter Notebook

## Network Architecture

| Layer  | Neurons | Activation |
| ------ | ------- | ---------- |
| Input  | 2       | None       |
| Hidden | 3       | Sigmoid    |
| Output | 1       | Sigmoid    |

```
Input (2)  -->  Hidden (3)  -->  Output (1)
```

## Dataset (XOR Truth Table)

| Input 1 | Input 2 | Output |
| ------- | ------- | ------ |
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

## Training Parameters

| Parameter         | Value                   |
| ----------------- | ----------------------- |
| Learning rate     | 0.1                     |
| Epochs            | 10,000                  |
| Random seed       | 42                      |
| Weight init       | Uniform random [0, 1)   |
| Bias init         | Zeros                   |

## How It Works

1. **Forward propagation**
   - Hidden layer: `sigmoid(X · W_ih + b_h)`
   - Output layer: `sigmoid(H · W_ho + b_o)`
2. **Error calculation**
   - `error = y - output`
3. **Backpropagation**
   - Output delta: `error * sigmoid'(output)`
   - Hidden error: `delta_output · W_ho.T`
   - Hidden delta: `hidden_error * sigmoid'(hidden)`
4. **Weight and bias update** (gradient descent)
   - `W_ho += H.T · delta_output * learning_rate`
   - `b_o  += sum(delta_output) * learning_rate`
   - `W_ih += X.T · delta_hidden * learning_rate`
   - `b_h  += sum(delta_hidden) * learning_rate`

The sigmoid function and its derivative:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)   # x is already the sigmoid output
```

## Results

After 10,000 epochs, the network's predictions on the XOR inputs are:

| Input  | Expected | Predicted  |
| ------ | -------- | ---------- |
| [0, 0] | 0        | 0.0578     |
| [0, 1] | 1        | 0.9504     |
| [1, 0] | 1        | 0.9501     |
| [1, 1] | 0        | 0.0519     |

Rounding the outputs to the nearest integer gives `[0, 1, 1, 0]`, which matches the XOR truth table.

## Getting Started

### Requirements

- Python 3.x
- NumPy

```bash
pip install numpy
```

### Run

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Open the notebook:
   ```bash
   jupyter notebook ANN_Back_Propagation.ipynb
   ```
   Or upload it to [Google Colab](https://colab.research.google.com/).
3. Run all cells from top to bottom.

## Notebook Structure

1. **Setup:** imports, activation functions, network size
2. **Initialization:** weights, biases, training data, hyperparameters
3. **Training:** forward pass, backpropagation, weight updates
4. **Testing:** prediction on XOR inputs

## Key Concepts Covered

- Feed-forward neural network
- Sigmoid activation and its derivative
- Backpropagation and the chain rule
- Gradient descent weight updates
- Why XOR needs a hidden layer

## Possible Improvements

- Track and plot the loss over epochs
- Try different hidden layer sizes and learning rates
- Compare with other activations (ReLU, tanh)
- Re-implement with TensorFlow/Keras or PyTorch

## Author

**Tanzim**
