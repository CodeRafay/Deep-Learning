# **1. Feedforward Neural Networks (FNNs)**

### Definition

- A **Feedforward Neural Network** is the simplest form of Artificial Neural Network (ANN).
- The flow of information is **one-way only**: from inputs → hidden layers → outputs. No loops or feedback.

### Structure

- **Input Layer**: takes raw features (e.g., pixels of an image).
- **Hidden Layers**: each neuron applies a weighted sum + bias → activation function → passes to the next layer.
- **Output Layer**: final predictions (classification, regression, etc.).

### Mathematical Form

For one neuron:

$$
a = f(Wx + b)
$$

- $W$: weights,
- $x$: input vector,
- $b$: bias,
- $f$: activation function,
- $a$: neuron’s output.

### Characteristics

- **Universal function approximator**: With enough neurons/layers, FNNs can approximate any continuous function.
- **Static model**: no internal memory of past inputs. (In contrast: RNNs handle sequential data).
- **Training**: done using **backpropagation** + optimization algorithm (SGD, Adam, etc.).

### Related Concepts

- **Shallow vs Deep Networks**: one hidden layer vs many hidden layers.
- **Overfitting risk**: if network is too big without regularization.
- **Batch Normalization, Dropout**: techniques to improve training.
- **Vanishing/Exploding gradients**: problem with deep FNNs, led to innovations like ReLU and ResNets.

---

# **2. Motivation for Neural Networks: Need for Non-Linear Models**

### Why not Linear Models?

- **Linear models**: $y = w^Tx + b$.

  - Good for linearly separable data (like classifying if a point is above/below a line).
  - But fail for **complex patterns** (e.g., XOR problem, image recognition).

### Example: XOR Problem

- XOR is **not linearly separable**.
- A single line cannot divide (0,0), (1,1) vs (0,1), (1,0).
- Neural networks with **non-linear activation functions** solve XOR easily.

### Why Neural Networks?

- They **stack multiple layers** of neurons.
- Each layer performs a **non-linear transformation**, allowing composition of simple features into complex ones.
- Example: In image classification,

  - Lower layers detect edges,
  - Mid layers detect shapes,
  - Higher layers detect objects.

### Related Concepts

- **Non-linear decision boundaries**: curves, surfaces, etc.
- **Kernel trick in SVMs**: another way to handle non-linearity, but neural networks scale better with data.
- **Deep Learning revolution**: GPUs + big data allowed training of very large non-linear models.

---

# **3. Neural Network Architecture**

### General Layout

- **Input Layer**: number of neurons = number of features.
- **Hidden Layers**: can be 1 or many. Depth gives more power.
- **Output Layer**: depends on task.

### Parameters

- **Weights**: strength of connections between neurons.
- **Biases**: allow shifting the activation threshold.
- **Activation functions**: introduce non-linearity.
- **Loss function**: measures error (cross-entropy, MSE, etc.).

### Training Procedure

1. Forward Pass: compute outputs layer by layer.
2. Loss Calculation: compare prediction vs ground truth.
3. Backpropagation: compute gradients of loss wrt weights.
4. Weight Update: optimizer (SGD, Adam, RMSprop).

### Hyperparameters

- **Learning rate** (too high → unstable, too low → slow).
- **Batch size** (trade-off between speed and stability).
- **Number of layers/neurons** (capacity of model).

### Related Concepts

- **Overfitting vs Underfitting**:

  - Overfitting = memorizing training data,
  - Underfitting = failing to learn patterns.

- **Regularization**: L1/L2, Dropout, Early stopping.
- **Weight Initialization**: Xavier, He, etc. help training stability.

---

# **4. Neural Network Architecture: Activation Functions**

### Purpose

- Without activation functions, the entire network is just a **linear model** regardless of depth.
- Non-linear activations allow complex mappings.

### Common Activation Functions

1. **Sigmoid (Logistic)**

   $$
   f(x) = \frac{1}{1 + e^{-x}}
   $$
`f(x) = 1 / (1 + e^(-x))`

   - Outputs between (0,1).
   - Good for probability interpretation.
   - Problem: **Vanishing gradients** for large |x|.

3. **Tanh**

   $$
   f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
   $$
`f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

   - Outputs between (-1,1).
   - Centered at 0 (better than sigmoid).
   - Still suffers from vanishing gradient.

4. **ReLU (Rectified Linear Unit)**

   $$
   f(x) = \max(0, x)
   $$
`f(x) = max(0, x)`

   - Most widely used.
   - Pros: no vanishing gradient for positive inputs, very fast.
   - Cons: **Dying ReLU problem** (neurons stuck at 0 if weights misaligned).

5. **Leaky ReLU**

   $$
   f(x) = \max(0.01x, x)
   $$
`f(x) = max(0.01 * x, x)`

   - Fixes dying ReLU.

6. **ELU / GELU / Swish**

   - Advanced activations with smoother curves.
   - Used in modern architectures (Transformers often use GELU).

### Which is Best?

- **ReLU**: default choice for hidden layers.
- **Tanh/Sigmoid**: rarely used in hidden layers today, but still used in RNNs (e.g., LSTM gates).
- **GELU**: cutting-edge models (BERT, GPT) use it.

### Related Concepts

- **Derivative of activation**: must be easy to compute for backpropagation.
- **Batch Normalization**: reduces dependency on activation scaling.

---

# **5. Neural Network Architecture: Output Units (Softmax Activation Function)**

### Why Different Output Units?

The output layer’s activation depends on **type of problem**:

- **Regression**: Linear activation (no activation at output).
- **Binary Classification**: Sigmoid (output in \[0,1]).
- **Multiclass Classification**: Softmax.

### Softmax Function

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

- Converts raw scores (logits) into probabilities that sum to 1.
- Each class gets a probability.

### Why Softmax?

- Intuitive: outputs are probabilities.
- Ensures competition: increasing one probability decreases others.
- Works well with **Cross-Entropy Loss**, which is standard for classification.

### Related Concepts

- **Cross-Entropy Loss**:

  - For true label y and predicted softmax p:
 <!-- 
  $$
  L = -\sum y_i \log(p_i)
  $$
   -->
   
  `L = -∑ yᵢ log(pᵢ)`

  - Encourages correct class probability → 1.

- **One-Hot Encoding**: how labels are represented for softmax training.

- **Logits**: raw outputs before softmax.

### Alternatives

- **Sigmoid for Multi-label Classification**:

  - Each class independent, not mutually exclusive.

- **Linear for Regression**: predicts continuous values.

---




