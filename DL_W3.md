# **1. Loss Functions and Cost Functions**

### What is a **Loss Function**?

- A **loss function** measures how far a model’s prediction is from the actual (ground truth) value **for a single data point**.
- It’s basically an error metric: the smaller the loss, the better the prediction.
- Example:

  - Predicted = 0.8, Actual = 1.0
  - Squared Error = (0.8 − 1.0)² = 0.04 → this is the loss for that one point.

### What is a **Cost Function**?

- A **cost function** is the **average (or sum) of losses over the entire dataset**.
- So while loss = error for one training sample, cost = overall measure of how well the model is doing.

Formally,

$$
\text{Cost} = \frac{1}{N}\sum_{i=1}^{N} \text{Loss}(y_i, \hat{y}_i)
$$

### Why are Loss/Cost Functions Important?

- **Training Guidance**: They tell the optimizer (SGD, Adam, etc.) how to update weights.
- **Model Evaluation**: They measure progress during training and validation.
- **Choosing the Right Function**: Different tasks need different loss functions (regression vs classification).

### Where are they used?

- **In Backpropagation**: Gradients are computed by differentiating the loss function.
- **In Evaluation**: Even after training, cost functions are used to compare models.

---

# **2. Common Loss Functions**

## **A. Mean Squared Error (MSE)**

### Formula

$$
MSE = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

### Usage

- Standard for **regression problems**.

### Characteristics

- Penalizes larger errors more heavily (squared term).
- Always non-negative.
- Differentiable (good for optimization).

### Pros

- Smooth, convex, easy to differentiate.
- Works well when errors are normally distributed.

### Cons

- Sensitive to outliers (because squaring magnifies large errors).

---

## **B. Binary Cross-Entropy Loss (a.k.a. Log Loss)**

### Formula

$$
L = - \big[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \big]
$$

- $y \in \{0,1\}$ is the true label.
- $\hat{y} \in [0,1]$ is the predicted probability.

### Usage

- **Binary classification** problems (spam/not spam, yes/no).
- Works with **sigmoid output activation**.

### Characteristics

- Strongly penalizes confident but wrong predictions.
- If model predicts 0.99 but true label = 0 → very high loss.

### Pros

- Probabilistic interpretation.
- Encourages the model to output probabilities close to the true label.

---

## **C. Categorical Cross-Entropy Loss**

### Formula

For **K-class classification**:

$$
L = -\sum_{i=1}^K y_i \log(\hat{y}_i)
$$

- $y_i$: one-hot encoded true label.
- $\hat{y}_i$: predicted probability from **Softmax output**.

### Usage

- **Multiclass classification** (MNIST digit recognition, ImageNet classification).
- Each sample belongs to **one class only**.

### Characteristics

- Encourages the correct class probability → 1.
- Works hand-in-hand with **softmax** activation in the output layer.

---

# **3. Other Loss Functions**

Here are additional ones that may come in exams or practical use:

---

### **Huber Loss**

$$
L =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| < \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

- Combines **MSE** and **MAE (Mean Absolute Error)**.
- Less sensitive to outliers than MSE.
- Used in regression when data may have noise.

---

### **Hinge Loss**

$$
L = \max(0, 1 - y \cdot \hat{y})
$$

- Used for **SVMs** and sometimes neural networks.
- Good for classification problems.

---

### **Kullback-Leibler (KL) Divergence**

$$
D_{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}
$$

- Measures how one probability distribution differs from another.
- Used in **variational autoencoders (VAEs)**, **regularization**, and **information theory**.

---

### **Mean Absolute Error (MAE)**

$$
MAE = \frac{1}{N}\sum |y_i - \hat{y}_i|
$$

- Simpler than MSE.
- Robust to outliers, but less smooth for optimization.

---

### **Cross-Entropy with Label Smoothing**

- Instead of strict one-hot labels, assign small probabilities to other classes.
- Helps prevent **overconfidence** in predictions.

---

# **4. Quick Comparison**

| Loss Function                 | Typical Use               | Activation at Output | Notes                                 |
| ----------------------------- | ------------------------- | -------------------- | ------------------------------------- |
| **MSE**                       | Regression                | Linear               | Sensitive to outliers                 |
| **MAE**                       | Regression                | Linear               | More robust than MSE                  |
| **Huber**                     | Regression                | Linear               | Balance between MSE & MAE             |
| **Binary Cross-Entropy**      | Binary classification     | Sigmoid              | Penalizes wrong confident predictions |
| **Categorical Cross-Entropy** | Multiclass classification | Softmax              | Standard for classification           |
| **Hinge Loss**                | Classification (SVM)      | Linear               | Margin-based                          |
| **KL Divergence**             | Probabilistic models      | Softmax/Prob dists   | Used in VAEs, NLP, etc.               |

---

Summary :

- **Loss = single point error**, **Cost = overall error**.
- Pick **MSE/MAE/Huber** for regression, **Cross-Entropy** for classification.
- Loss functions are at the heart of backpropagation.

---

# **5. Optimization Basics**

### What is Optimization in Neural Networks?

- Once we have a **loss function** (measuring error), we need a way to **minimize it** by updating weights.
- This is done by an **optimization algorithm**.
- Core idea: adjust parameters in the direction that reduces loss the most.

### Key Concepts

- **Parameters**: weights $W$ and biases $b$.
- **Gradient**: slope/derivative of the loss function with respect to parameters.

  - Tells us how to change weights to reduce error.

- **Learning Rate ($\eta$)**: step size for weight updates.

  - Too high → divergence (overshooting).
  - Too low → very slow learning.

### Update Rule (General Form)

$$
\theta := \theta - \eta \cdot \nabla_\theta J(\theta)
$$

- $\theta$ = parameters (weights, biases).
- $\eta$ = learning rate.
- $J(\theta)$ = cost function.
- $\nabla_\theta J(\theta)$ = gradient of cost wrt parameters.

---

# **6. Gradient Descent (GD)**

### Vanilla Gradient Descent (Batch GD)

- Computes gradient using **entire dataset**.

$$
\theta := \theta - \eta \cdot \nabla_\theta J(\theta)
$$

- Guarantees smooth convergence for convex problems.

#### Pros

- Stable convergence.
- True direction of steepest descent.

#### Cons

- Very slow for large datasets (need to process all samples before one update).
- Memory expensive.

---

# **7. Stochastic Gradient Descent (SGD)**

### Definition

- Instead of using the whole dataset, update weights using **one sample at a time**.

$$
\theta := \theta - \eta \cdot \nabla_\theta J(\theta; x^{(i)}, y^{(i)})
$$

### Characteristics

- Updates are noisy, but this noise can help escape local minima.
- Converges faster than batch GD (frequent updates).

#### Pros

- Fast updates (especially for large datasets).
- Helps in escaping shallow local minima.

#### Cons

- Very noisy trajectory → loss curve fluctuates heavily.
- Requires tuning learning rate carefully.

### Related Improvements

- **Momentum**: smooths updates by adding a velocity term.
- **Adaptive learning rates**: AdaGrad, RMSprop, Adam.

---

# **8. Mini-Batch Gradient Descent**

### Definition

- Compromise between Batch GD and SGD.
- Uses **a small subset of data (mini-batch)** for each update.

$$
\theta := \theta - \eta \cdot \nabla_\theta J(\theta; x^{(i:i+m)}, y^{(i:i+m)})
$$

- Typical batch sizes: 16, 32, 64, 128.

### Characteristics

- More stable than SGD.
- Faster than full Batch GD.
- Works well with GPU parallelization.

#### Pros

- Efficiency + stability.
- Smooth convergence but still allows some stochasticity.

#### Cons

- Choosing batch size is tricky (too small → noisy, too large → slow).

---

# **9. Related Concepts (Exam-Relevant)(will be covered in next, no worries)**

### Momentum

- Adds a fraction of the previous update to the current one.
- Prevents oscillations and accelerates convergence.

$$
v_t = \beta v_{t-1} + \eta \nabla_\theta J(\theta)
$$

$$
\theta := \theta - v_t
$$

### Nesterov Accelerated Gradient (NAG)

- Looks ahead by applying momentum before computing gradient.
- Improves convergence speed.

### Adaptive Optimizers

- **AdaGrad**: adapts learning rate per parameter (good for sparse data, NLP).
- **RMSprop**: scales updates by moving average of squared gradients.
- **Adam (Adaptive Moment Estimation)**: combines momentum + RMSprop.

  - Most popular today.

### Learning Rate Scheduling

- **Decay**: gradually reduce learning rate during training.
- **Warm restarts**: periodically reset learning rate for better exploration.

---

# **10. Backpropagation**

## **Concept of Backpropagation**

- **Goal**: Train a neural network by minimizing the loss function.
- Neural networks are **composed of layers of functions**. Each function’s output depends on weights and inputs.
- To update weights using **gradient descent**, we need gradients of the loss wrt weights.
- Backpropagation uses the **chain rule of calculus** to efficiently compute these gradients from output layer → backward to input layer.

**Flow:**

1. **Forward Pass**: compute predictions step by step.
2. **Loss Calculation**: measure error with a loss function.
3. **Backward Pass (Backpropagation)**: propagate error backward using chain rule to compute gradients.
4. **Parameter Update**: apply gradient descent (or its variants).

---

## **Mathematics of Backpropagation**

### General Formula

For a neuron:

$$
z = w \cdot x + b
$$

$$
a = f(z)
$$

- $z$ = linear combination of inputs
- $f(z)$ = activation function
- $a$ = neuron output

During backpropagation, we compute:

$$
\frac{\partial L}{\partial w}, \quad \frac{\partial L}{\partial b}, \quad \frac{\partial L}{\partial x}
$$

using the chain rule:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

This cascades backward through layers.

---

## **Worked Example: A Tiny Neural Network**

Let’s use a **2–2–1 network** (2 inputs, 1 hidden layer with 2 neurons, 1 output).
Task: **binary classification** (sigmoid output).

---

### Step 1: Define the Network

- Inputs: $x_1, x_2$
- Hidden Layer (2 neurons, activation = sigmoid):

  - $z_1 = w_{11}x_1 + w_{21}x_2 + b_1$
  - $a_1 = \sigma(z_1)$
  - $z_2 = w_{12}x_1 + w_{22}x_2 + b_2$
  - $a_2 = \sigma(z_2)$

- Output Layer (1 neuron, sigmoid):

  - $z_3 = w_{13}a_1 + w_{23}a_2 + b_3$
  - $\hat{y} = \sigma(z_3)$

Loss function: Binary Cross-Entropy (BCE).

---

### Step 2: Example Numbers

- Input: $x = [1, 0]$
- True label: $y = 1$
- Initial weights & biases:

  - $w_{11} = 0.2, w_{21} = 0.4, b_1 = 0.1$
  - $w_{12} = 0.3, w_{22} = 0.1, b_2 = 0.2$
  - $w_{13} = 0.7, w_{23} = 0.5, b_3 = 0.3$

- Activation: Sigmoid

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

---

### Step 3: Forward Pass

- Hidden neuron 1:

  $$
  z_1 = (0.2)(1) + (0.4)(0) + 0.1 = 0.3
  $$

  $$
  a_1 = \sigma(0.3) \approx 0.574
  $$

- Hidden neuron 2:

  $$
  z_2 = (0.3)(1) + (0.1)(0) + 0.2 = 0.5
  $$

  $$
  a_2 = \sigma(0.5) \approx 0.622
  $$

- Output neuron:

  $$
  z_3 = (0.7)(0.574) + (0.5)(0.622) + 0.3 \approx 1.086
  $$

  $$
  \hat{y} = \sigma(1.086) \approx 0.748
  $$

So the model predicts $ \hat{y} \approx 0.748$.

---

### Step 4: Loss Calculation

Binary Cross-Entropy Loss:

$$
L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

Since $y=1$:

$$
L = -\log(0.748) \approx 0.289
$$

---

### Step 5: Backpropagation (Gradients)

#### Output Layer

Derivative of BCE loss wrt output neuron:

$$
\delta_3 = \hat{y} - y = 0.748 - 1 = -0.252
$$

Gradients for output weights:

$$
\frac{\partial L}{\partial w_{13}} = \delta_3 \cdot a_1 = (-0.252)(0.574) \approx -0.145
$$

$$
\frac{\partial L}{\partial w_{23}} = \delta_3 \cdot a_2 = (-0.252)(0.622) \approx -0.157
$$

$$
\frac{\partial L}{\partial b_3} = \delta_3 = -0.252
$$

---

#### Hidden Layer

For hidden neuron 1:

$$
\delta_1 = (w_{13}\delta_3)\cdot \sigma'(z_1)
$$

$$
\sigma'(z) = \sigma(z)(1-\sigma(z))
$$

- $\sigma'(0.3) = 0.574(1-0.574) = 0.244$
- $\delta_1 = (0.7)(-0.252)(0.244) \approx -0.043$

For hidden neuron 2:

- $\sigma'(0.5) = 0.622(1-0.622) = 0.235$
- $\delta_2 = (0.5)(-0.252)(0.235) \approx -0.030$

Gradients for hidden weights:

$$
\frac{\partial L}{\partial w_{11}} = \delta_1 \cdot x_1 = (-0.043)(1) = -0.043
$$

$$
\frac{\partial L}{\partial w_{21}} = \delta_1 \cdot x_2 = (-0.043)(0) = 0
$$

$$
\frac{\partial L}{\partial b_1} = \delta_1 = -0.043
$$

$$
\frac{\partial L}{\partial w_{12}} = \delta_2 \cdot x_1 = (-0.030)(1) = -0.030
$$

$$
\frac{\partial L}{\partial w_{22}} = \delta_2 \cdot x_2 = (-0.030)(0) = 0
$$

$$
\frac{\partial L}{\partial b_2} = \delta_2 = -0.030
$$

---

### Step 6: Weight Update (Gradient Descent)

Learning rate: $\eta = 0.1$.

Example update for $w_{13}$:

$$
w_{13}^{new} = w_{13} - \eta \cdot \frac{\partial L}{\partial w_{13}}
= 0.7 - 0.1(-0.145) = 0.7145
$$

Do similar updates for all weights and biases.

---

### Step 7: Repeat

- Next iteration: forward pass with updated weights → new loss → backprop again.
- Over many epochs, loss decreases, predictions improve.

---

# **11. Summary of Backpropagation Cycle**

1. **Forward pass**: compute activations layer by layer.
2. **Loss computation**: compare prediction with true label.
3. **Backward pass**: compute gradients using chain rule.
4. **Parameter update**: use Gradient Descent (SGD, Mini-batch, Adam, etc.).
5. Repeat until convergence.

---

This small worked example showed:  
 **input layer → hidden activations → output → loss → gradient descent → backpropagation**.

---
