# Deep Learning Assignment 2

## Overview
This assignment demonstrates two fundamental approaches to deep learning: building neural networks from scratch and using modern frameworks with regularization techniques. The project consists of two main parts implemented in Jupyter notebooks.

---

## Part A: Multilayer Perceptron (MLP) from Scratch

[1AM22CI109_TanishqRU_PartA.ipynb](1AM22CI109_TanishqRU_PartA.ipynb)

### Objective
Implement a fully-connected neural network from scratch using only NumPy to classify MNIST handwritten digits.

### Implementation Details

**Architecture:**
- Input Layer: 784 neurons (28×28 flattened images)
- Hidden Layer 1: 128 neurons (ReLU activation)
- Hidden Layer 2: 64 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax activation for multi-class classification)

**Key Components:**
1. **Activation Functions:**
   - ReLU (Rectified Linear Unit) for hidden layers
   - Softmax for output layer (multi-class probability distribution)

2. **Forward Propagation:**
   - Computes Z = WX + b and applies activation functions
   - Caches intermediate values for backward pass

3. **Backward Propagation:**
   - Computes gradients using chain rule
   - Updates weights and biases using gradient descent

4. **Initialization:**
   - He Initialization: W ~ N(0, 2/n_in) for stable training
   - Bias vectors initialized to zero

5. **Training:**
   - Mini-batch gradient descent with batch size 64
   - Learning rate: 0.1
   - Epochs: 25
   - Cross-entropy loss function

**Key Features:**
- Data normalization to [0, 1]
- One-hot encoding for labels
- Epoch-wise loss tracking
- Metrics: accuracy, confusion matrix, classification report

### Output
- Training loss curve showing convergence
- Test accuracy on 10,000 unseen MNIST samples
- Detailed classification report with precision, recall, F1-score
- Confusion matrix heatmap

---

## Part B: CNN with Regularization (TensorFlow/Keras)

[1AM22CI109_TanishqRU_PartB.ipynb](1AM22CI109_TanishqRU_PartB.ipynb)

### Objective
Build a Convolutional Neural Network with regularization techniques (data augmentation, dropout, L2 regularization) to classify Fashion MNIST images, comparing performance with and without regularization.

### Architecture

**Base Model:**
- **Conv Block 1:** 32 filters, (3×3) kernel, ReLU activation + Max Pooling (2×2)
- **Conv Block 2:** 64 filters, (3×3) kernel, ReLU activation + Max Pooling (2×2)
- **Conv Block 3:** 128 filters, (3×3) kernel, ReLU activation + Max Pooling (2×2)
- **Flatten** → Dense layers → Softmax output

**Dense Layers:**
- 256 neurons (ReLU) with optional L2 regularization
- 128 neurons (ReLU) with optional L2 regularization
- 10 neurons (Softmax) for classification

### Regularization Techniques

**Experiment 1: Baseline (No Regularization)**
- Standard CNN without augmentation or dropout
- Baseline for performance comparison

**Experiment 2: Regularized (Full Regularization)**
1. **Data Augmentation:**
   - Random horizontal flips
   - Random rotations (±4 degrees / ~15 degrees)
   - Random zoom (±10%)

2. **Dropout:**
   - 50% dropout after first dense layer
   - 30% dropout after second dense layer

3. **L2 Regularization:**
   - L2 penalty (λ=0.001) on dense layer weights
   - Prevents overfitting by penalizing large weights

4. **Learning Rate Scheduling:**
   - ReduceLROnPlateau: reduces learning rate when validation loss plateaus

5. **Early Stopping:**
   - Patience of 5 epochs; restores best weights

### Training Configuration
- Optimizer: Adam
- Loss: Categorical cross-entropy
- Batch size: 64
- Epochs: 30 (with early stopping)
- Validation split: 20%
- Dataset: Fashion MNIST (60,000 train, 10,000 test)

### Output
- Comparison plots: validation accuracy and loss (Baseline vs. Regularized)
- Test accuracy for both models
- Demonstrates regularization's impact on generalization

---

## Dataset Information

### Part A: MNIST
- **Classes:** 10 (digits 0-9)
- **Training samples:** 60,000
- **Test samples:** 10,000
- **Image size:** 28×28 grayscale

### Part B: Fashion MNIST
- **Classes:** 10 (clothing items: T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)
- **Training samples:** 60,000
- **Test samples:** 10,000
- **Image size:** 28×28 grayscale

---

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Installation
```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

---

## Files

- **1AM22CI109_TanishqRU_PartA.ipynb** - MLP implementation from scratch using NumPy
- **1AM22CI109_TanishqRU_PartB.ipynb** - CNN with regularization techniques using TensorFlow
- **README.md** - This file

---

## How to Run

### Part A: MLP from Scratch
1. Open [`1AM22CI109_TanishqRU_PartA.ipynb`](1AM22CI109_TanishqRU_PartA.ipynb) in Jupyter
2. Run all cells sequentially
3. Monitor training loss and epoch progress
4. View final test accuracy, confusion matrix, and classification report

**Note:** Part A uses pure NumPy and may take several minutes to train (no GPU acceleration).

### Part B: CNN with Regularization
1. Open [`1AM22CI109_TanishqRU_PartB.ipynb`](1AM22CI109_TanishqRU_PartB.ipynb) in Jupyter
2. Run all cells sequentially
3. Two models will train automatically: baseline and regularized
4. Compare validation metrics in output plots

**Tip:** Use GPU if available for faster training (TensorFlow will auto-detect).

---

## Key Learnings

### Part A
- Understanding neural network mathematics from first principles
- Forward and backward propagation implementation
- Weight initialization strategies (He initialization)
- Mini-batch gradient descent
- Building intuition about hyperparameter choices

### Part B
- Modern deep learning frameworks (TensorFlow/Keras)
- Convolutional layers and their benefits for image data
- Regularization techniques to prevent overfitting:
  - **Data augmentation:** increases dataset diversity
  - **Dropout:** random neuron deactivation during training
  - **L2 regularization:** weight decay penalty
  - **Early stopping:** prevents training beyond optimal point
- Comparing model performance with/without regularization

---

## Expected Results

### Part A (MLP from Scratch)
- **Test Accuracy:** ~97-98% on MNIST
- Training demonstrates convergence through loss curve
- Confusion matrix shows strong diagonal (correct classifications)

### Part B (CNN with Regularization)
- **Baseline Test Accuracy:** ~91-93%
- **Regularized Test Accuracy:** ~93-95%
- Regularized model shows better generalization (lower gap between train and validation loss)

---

## Author
1AM22CI109 - Tanishq RU

---

## Notes
- All datasets are automatically downloaded from TensorFlow/Keras upon first run
- Random seeds are set for reproducibility
- Adjust hyperparameters (learning rate, batch size, epochs) for different results
- GPU support in Part B significantly speeds up training