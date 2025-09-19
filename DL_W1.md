# 1. Artificial Intelligence (AI)

### What is AI?

- **AI** is the broadest concept. It refers to the **simulation of human intelligence** in machines that are programmed to think, learn, and solve problems.
- AI aims to create systems capable of performing tasks that normally require human intelligence, such as reasoning, planning, understanding natural language, vision, decision making, etc.

### Scope:

- AI includes any technique or method that enables machines to mimic human cognitive functions.
- Can be **rule-based systems** (expert systems), symbolic logic, search algorithms, robotics, and more.

### Examples:

- Chess-playing programs.
- Voice assistants (Siri, Alexa).
- Autonomous vehicles.
- Spam filters.

### Summary:

- AI = the overall science of mimicking human intelligence.
- Encompasses all subfields including ML and DL.

---

# 2. Machine Learning (ML)

### What is ML?

- ML is a **subset of AI** focused specifically on the idea that machines can **learn from data** and improve from experience **without being explicitly programmed** with specific rules.
- Instead of hard-coding instructions, ML algorithms identify patterns in data to make predictions or decisions.

### How ML works:

- You provide data (input and, optionally, labels).
- The algorithm builds a model by learning patterns from data.
- The model makes predictions on new, unseen data.

### Types of ML:

- **Supervised learning:** learns from labeled data (input-output pairs).
- **Unsupervised learning:** finds patterns in unlabeled data.
- **Reinforcement learning:** learns by interacting with the environment to maximize reward.

### Examples:

- Email spam detection.
- Fraud detection.
- Recommendation systems.
- Image classification.

### Summary:

- ML = algorithms that learn from data to perform tasks.
- Subset of AI focused on data-driven learning.

---

# 3. Deep Learning (DL)

### What is DL?

- DL is a **subset of machine learning** that uses **artificial neural networks** with many layers (hence “deep”) to model complex patterns in large amounts of data.
- It’s inspired by the structure and function of the brain’s neural networks.

### Key features:

- Uses multi-layered neural networks (deep neural networks).
- Learns hierarchical feature representations (low-level to high-level abstractions).
- Requires large datasets and high computational power (GPUs).

### Examples:

- Image recognition (e.g., detecting objects in photos).
- Natural language processing (e.g., language translation, chatbots).
- Speech recognition.
- Autonomous driving perception.

### Summary:

- DL = deep neural networks learning from large data.
- Subset of ML with more complex models and more powerful representation learning.

---

## **Visualizing the relationship**

```
Artificial Intelligence (AI)
    └── Machine Learning (ML)
           └── Deep Learning (DL)
```

- DL is a specialized type of ML.
- ML is one way to achieve AI.
- AI also includes other approaches beyond ML (like rule-based systems).

---

## **Key differences**

| Aspect           | AI                                           | ML                                           | DL                                                        |
| ---------------- | -------------------------------------------- | -------------------------------------------- | --------------------------------------------------------- |
| Definition       | Broad science of making machines intelligent | Algorithms that learn from data              | Neural networks with many layers learning from large data |
| Focus            | Mimic human intelligence broadly             | Learn patterns from data                     | Automatically learn hierarchical features in data         |
| Techniques       | Rules, logic, search, ML, DL, etc.           | Regression, decision trees, SVM, neural nets | Deep neural networks, CNNs, RNNs                          |
| Data requirement | Can be manual rules-based or data-driven     | Needs data                                   | Needs large datasets                                      |
| Computation      | Can be simple or complex                     | Medium complexity                            | High computational power (GPUs often needed)              |
| Examples         | Expert systems, game AI, robotics            | Spam filters, recommendation systems         | Image/speech recognition, language translation            |
| Interpretability | Often interpretable (rules-based)            | Moderate                                     | Often “black-box” and harder to interpret                 |

---

### Summary in plain words:

- **AI** is the big umbrella, everything related to making machines smart.
- **ML** is a way machines get smart by learning from data.
- **DL** is a powerful kind of ML that uses deep neural networks to learn very complex patterns from massive data.

---

# Machine Learning (ML) Overview

- **Machine Learning** is a field of AI that gives computers the ability to learn patterns from data and make decisions or predictions.
- ML algorithms automatically improve their performance as they are exposed to more data.

---

## Types of Machine Learning

### 1. Supervised Learning

- The model learns from **labeled data** — each input example has a corresponding output label.
- The goal is to learn a mapping from inputs to outputs so the model can predict the output for new, unseen inputs.

#### Typical Tasks in Supervised Learning:

- **Classification:** Predict categorical labels.
- **Regression:** Predict continuous values.

---

### 2. Unsupervised Learning

- The model learns from **unlabeled data** — only input data is given, no output labels.
- The goal is to find structure, patterns, or relationships in the data.

#### Typical Tasks in Unsupervised Learning:

- **Clustering:** Group data into clusters.
- **Dimensionality Reduction:** Simplify data by reducing features.

---

## 1. Supervised Learning in Detail

---

### 3.1 Classification

- **Definition:** Predict a discrete class label from input features.
- **Output:** One of several categories (e.g., spam or not spam).
- **Examples:**

  - Email spam detection (spam/not spam).
  - Handwritten digit recognition (digits 0-9).
  - Disease diagnosis (disease present/absent).

### Popular Algorithms:

- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees and Random Forests
- Neural Networks

---

### 3.2 Regression

- **Definition:** Predict a continuous output value based on input features.
- **Output:** Real numbers (e.g., price, temperature).
- **Examples:**

  - Predicting house prices.
  - Forecasting stock prices.
  - Predicting temperature.

### Popular Algorithms:

- Linear Regression
- Polynomial Regression
- Support Vector Regression (SVR)
- Neural Networks

---

## 4. Unsupervised Learning in Detail

---

### 4.1 Clustering

- **Definition:** Group similar data points into clusters based on features.
- **Goal:** Discover natural groupings in data.
- **Examples:**

  - Customer segmentation for marketing.
  - Grouping news articles by topic.
  - Image segmentation.

### Popular Algorithms:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN (Density-Based Spatial Clustering)

---

### 4.2 Dimensionality Reduction

- **Definition:** Reduce the number of input features while preserving important information.
- **Goal:** Simplify data, visualize high-dimensional data.
- **Examples:**

  - Visualizing high-dimensional data in 2D or 3D.
  - Preprocessing before clustering or classification.

### Popular Algorithms:

- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Autoencoders (neural networks)

### 4.3 Association

- **Definition:** Discover interesting relationships or patterns between variables in large datasets.
- **Goal:** Find rules that describe likely co-occurrences of items or events.
- **Examples:**
  - Market basket analysis (what products are bought together).
  - Web usage patterns (which pages are visited together).
  - Medical symptom associations.

#### Popular Algorithms:

- Apriori Algorithm
- FP-Growth (Frequent Pattern Growth)
- ECLAT (Equivalence Class Transformation)

#### Key Concepts:

- **Support:** Frequency of item combinations
- **Confidence:** Probability of Y given X
- **Lift:** Strength of association between items

#### Common Applications:

- Retail store product placement
- Recommendation systems
- Cross-selling strategies
- Website navigation design
- Disease co-occurrence analysis

| Metric     | Formula                  | Description                        |
| ---------- | ------------------------ | ---------------------------------- |
| Support    | P(X ∩ Y)                 | How often items appear together    |
| Confidence | P(Y\|X)                  | How often Y occurs when X occurs   |
| Lift       | P(X ∩ Y) / (P(X) × P(Y)) | Independence measure between items |

---

## 5. Other ML Categories & Learning Types

---

## 5.1 Semi-Supervised Learning

- Uses a **small amount of labeled data** and a large amount of unlabeled data.
- Useful when labeling is expensive or time-consuming.
- Combines supervised and unsupervised learning techniques.

---

## 5.2 Reinforcement Learning

- An agent learns to make decisions by interacting with an environment.
- Learns through rewards or punishments (feedback).
- Used in robotics, game playing, autonomous driving.

---

## 6. Summary Table

| Learning Type                | Data Used                       | Task Types                                        | Goal                                 | Examples                         |
| ---------------------------- | ------------------------------- | ------------------------------------------------- | ------------------------------------ | -------------------------------- |
| **Supervised Learning**      | Labeled data                    | Classification, Regression                        | Predict outputs from inputs          | Spam detection, price prediction |
| **Unsupervised Learning**    | Unlabeled data                  | Clustering, Dimensionality Reduction, Association | Find patterns/groups in data         | Customer segmentation, PCA       |
| **Semi-Supervised Learning** | Small labeled + large unlabeled | Mix of supervised & unsupervised                  | Improve learning with limited labels | Web page classification          |
| **Reinforcement Learning**   | Environment feedback            | Policy learning, decision making                  | Maximize cumulative reward           | Game AI, robotics                |

---

# Why Deep Learning?

---

## 1. **Ability to Learn Complex, Hierarchical Features Automatically**

- Traditional ML models often require **manual feature engineering**: domain experts need to handcraft features from raw data (e.g., edges, textures in images).
- **Deep learning** uses **deep neural networks** with many layers that automatically learn **hierarchical representations**:

  - Lower layers learn simple features (edges, colors).
  - Higher layers learn complex concepts (faces, objects, emotions).

- This **automatic feature extraction** reduces the need for manual intervention and can discover subtle patterns humans might miss.

---

## 2. **Handles Large, High-Dimensional Data**

- Modern datasets (images, audio, text, video) are **large-scale and high-dimensional**.
- Deep networks are capable of modeling very **complex functions** in these large feature spaces.
- Traditional algorithms struggle with high dimensionality or require dimensionality reduction beforehand.
- Deep learning naturally handles raw inputs without heavy preprocessing.

---

## 3. **State-of-the-Art Performance on Many Tasks**

- Deep learning has achieved **significant breakthroughs** in fields such as:

  - **Computer Vision:** Image recognition, object detection, segmentation.
  - **Natural Language Processing (NLP):** Translation, summarization, chatbots.
  - **Speech Recognition:** Transcribing spoken language.
  - **Game Playing:** Beating human champions (e.g., AlphaGo).

- Many benchmarks and competitions show deep learning outperforms classical ML methods.

---

## 4. **End-to-End Learning**

- Deep learning models can be trained **end-to-end**, directly from raw input data to the final output.
- This contrasts with traditional ML pipelines which may require multiple disconnected stages (feature extraction, dimensionality reduction, classification).
- End-to-end models simplify workflows and reduce error propagation.

---

## 5. **Flexibility and Generality**

- Deep learning architectures can be adapted for various data types:

  - **Convolutional Neural Networks (CNNs)** for images and spatial data.
  - **Recurrent Neural Networks (RNNs) and Transformers** for sequential and language data.
  - **Autoencoders** for unsupervised learning and dimensionality reduction.

- This versatility makes deep learning applicable across many domains.

---

## 6. **Improved with More Data and Compute**

- Deep learning models tend to **improve with more data**.
- The availability of **large datasets** and powerful hardware (GPUs, TPUs) has fueled deep learning’s success.
- Many traditional ML models saturate after a certain point, while deep learning can keep improving.

---

## 7. **Robustness to Noise and Variability**

- Deep networks can be more robust to variations and noise in data due to learned representations.
- Techniques like **dropout**, **batch normalization**, and **data augmentation** improve generalization.

---

## 8. **Challenges Deep Learning Addresses Better Than Traditional ML**

| Challenge                        | Traditional ML                      | Deep Learning                           |
| -------------------------------- | ----------------------------------- | --------------------------------------- |
| Feature Engineering              | Requires manual design              | Learns features automatically           |
| Large-scale data                 | May struggle or need simplification | Scales well with big data               |
| Complex patterns                 | Limited expressiveness              | Can model complex hierarchical features |
| Unstructured data (images, text) | Requires heavy preprocessing        | Handles raw data end-to-end             |
| Multimodal data                  | Difficult to integrate              | Can integrate multiple data types       |

---

## Summary: Why Deep Learning?

| Reason                             | Explanation                                                 |
| ---------------------------------- | ----------------------------------------------------------- |
| **Automatic Feature Learning**     | Learns complex features from raw data without manual design |
| **High-dimensional data handling** | Works well with images, audio, text, video                  |
| **Superior performance**           | Leads benchmarks in many AI tasks                           |
| **End-to-end training**            | Simplifies pipelines                                        |
| **Versatility**                    | Adaptable architectures for different data types            |
| **Data and compute scalability**   | Performance improves with more data and hardware            |
| **Robustness**                     | More resilient to noise and variability                     |

---

# The Perceptron:

---

## 1. What is a Perceptron?

- A **perceptron** is a simple **binary classifier** that maps input features to a binary output (usually 0 or 1).
- It’s a fundamental building block of neural networks.
- Designed to separate data points into two classes using a **linear decision boundary**.

---

## 2. Perceptron Structure

- Inputs: $\mathbf{x} = [x_1, x_2, ..., x_n]$
- Weights: $\mathbf{w} = [w_1, w_2, ..., w_n]$
- Bias: $b$ (sometimes considered $w_0$ with input $x_0=1$)
- Weighted sum: $z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^n w_i x_i + b$
- Activation function (Step function):

$$
f(z) = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

---

## 3. Perceptron Operation

- For each input vector $\mathbf{x}$, compute $z = \mathbf{w} \cdot \mathbf{x} + b$.
- Apply the step function to $z$ to get output $y$.
- The output predicts the class label (0 or 1).

---

## 4. Perceptron Learning Algorithm

The goal of the perceptron learning algorithm is to find the weights $\mathbf{w}$ and bias $b$ that correctly classify the training data if it’s linearly separable.

### Given:

- Training dataset $\{(\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), ..., (\mathbf{x}^{(m)}, y^{(m)})\}$

  - $\mathbf{x}^{(i)}$ is the input vector for the $i^{th}$ example.
  - $y^{(i)} \in \{0,1\}$ is the true label.

### Steps:

1. **Initialize weights and bias** to small random values or zeros:

$$
w_i = 0, \quad b = 0 \quad \forall i
$$

2. **For each training example** $(\mathbf{x}^{(i)}, y^{(i)})$:

   a. Compute the predicted output $\hat{y}$:

$$
z = \mathbf{w} \cdot \mathbf{x}^{(i)} + b
$$

$$
\hat{y} = f(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}
$$

b. Calculate the **error**:

$$
\text{error} = y^{(i)} - \hat{y}
$$

c. **Update the weights and bias** if the prediction is wrong (error $\neq 0$):

$$
w_j \leftarrow w_j + \eta \times \text{error} \times x_j^{(i)} \quad \text{for } j=1,...,n
$$

$$
b \leftarrow b + \eta \times \text{error}
$$

Here, $\eta$ is the **learning rate**, a small positive constant (e.g., 0.1).

3. Repeat Step 2 for all training examples **until**:

- The algorithm converges (no errors on all examples), or
- A maximum number of iterations (epochs) is reached.

---

## 5. Intuition Behind the Weight Update Rule

- If the perceptron **correctly classifies** the example, no update occurs.
- If the perceptron **misclassifies** an example, weights are updated to reduce future error:

  - If predicted output is too low (predicted 0 but true label 1), increase weights on active features.
  - If predicted output is too high (predicted 1 but true label 0), decrease weights on active features.

- Bias updated similarly to shift the decision boundary.

---

## 6. Example Walkthrough

Suppose:

- $\mathbf{x} = [2, 3]$, label $y=1$
- Initial weights $\mathbf{w} = [0, 0]$, bias $b=0$
- Learning rate $\eta = 0.1$

Step 1: Compute prediction

$$
z = 0*2 + 0*3 + 0 = 0 \Rightarrow \hat{y} = 1 \quad (\text{since } z \geq 0)
$$

Step 2: Error

$$
\text{error} = 1 - 1 = 0
$$

No update needed.

---

Now suppose another input:

- $\mathbf{x} = [1, 1]$, label $y=0$
- Current weights $\mathbf{w} = [0, 0]$, bias $b=0$

Step 1: Compute prediction

$$
z = 0*1 + 0*1 + 0 = 0 \Rightarrow \hat{y} = 1
$$

Step 2: Error

$$
\text{error} = 0 - 1 = -1
$$

Step 3: Update weights and bias:

$$
w_1 \leftarrow 0 + 0.1 \times (-1) \times 1 = -0.1
$$

$$
w_2 \leftarrow 0 + 0.1 \times (-1) \times 1 = -0.1
$$

$$
b \leftarrow 0 + 0.1 \times (-1) = -0.1
$$

---

## 7. Important Notes

- **Convergence:** The perceptron learning algorithm **converges only if the data is linearly separable**.
- **Learning Rate $\eta$:** Controls step size in updates. Too large can overshoot, too small slows learning.
- **Extension:** The perceptron can be extended to **multi-layer perceptrons (MLPs)** which can solve non-linear problems.

---

## 8. Summary Table: Perceptron Learning Algorithm

| Step                        | Description                                                                                                          |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Initialize weights and bias | Set weights $w_i$ and bias $b$ to zeros or small random values                                                       |
| Predict                     | Compute weighted sum $z = \mathbf{w} \cdot \mathbf{x} + b$, apply step function                                      |
| Calculate error             | $\text{error} = y - \hat{y}$                                                                                         |
| Update weights and bias     | $w_i \leftarrow w_i + \eta \times \text{error} \times x_i$, $b \leftarrow b + \eta \times \text{error}$ if error ≠ 0 |
| Repeat                      | Iterate over dataset until convergence or max iterations                                                             |

---

# Types of Activation Functions

## 1. Linear Activation Function

### Definition:

$$
f(x) = x
$$

### Explanation:

- Outputs input as-is.
- No non-linearity.
- Often used in output layers for **regression problems** where the target is continuous.

### Properties:

- Differentiable everywhere.
- Does **not** allow the network to model non-linear relationships if used in all layers.

### Pros:

- Simple, efficient.
- Allows for continuous output in regression tasks.

### Cons:

- Cannot capture non-linear patterns if used as the only activation.

### Use case:

- Output layer in regression tasks (predicting prices, temperatures, etc.).

---

## 2. Threshold (Step) Function

### Definition:

$$
f(x) = \begin{cases}
1 & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}
$$

### Explanation:

- Binary output 0 or 1.
- Used in original perceptron to decide whether a neuron fires.

### Properties:

- Non-differentiable at $x=0$.
- Produces **hard** binary decisions.

### Pros:

- Simple binary classifier.
- Intuitive threshold-based output.

### Cons:

- No gradient for training with gradient-based methods.
- Can only solve linearly separable problems.

### Use case:

- Original perceptrons and simple binary classification without gradient learning.

---

## 3. Sign Function

### Definition:

$$
f(x) = \begin{cases}
1 & \text{if } x \geq 0 \\
-1 & \text{if } x < 0
\end{cases}
$$

### Explanation:

- Outputs -1 or +1.
- Symmetric binary output useful when labels are coded as $\pm 1$.

### Properties:

- Non-differentiable.
- Hard binary output.

### Pros & Cons:

- Same as threshold function but symmetric outputs.

### Use case:

- Binary classification with $\pm 1$ labels.

---

## 4. Sigmoid Function

### Definition:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### Explanation:

- Outputs a smooth curve from 0 to 1.
- Converts any real-valued input into a probability-like output.

### Properties:

- Differentiable everywhere.
- Output range: (0, 1).
- Not zero-centered (outputs always positive).
- Gradients get very small for very positive or negative inputs (vanishing gradient problem).

### Pros:

- Good for probabilistic binary classification.
- Enables gradient-based optimization.

### Cons:

- Vanishing gradient slows training in deep networks.
- Outputs not zero-centered which can slow convergence.

### Use case:

- Output layers for binary classification.
- Sometimes hidden layers in shallow networks (though less common now).

---

## 5. Hyperbolic Tangent (tanh) Function

### Definition:

$$
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

### Explanation:

- Similar shape to sigmoid but outputs between -1 and 1.
- Zero-centered output helps with optimization.

### Properties:

- Differentiable.
- Output range: (-1, 1).
- Steeper gradient near zero than sigmoid.

### Pros:

- Zero-centered outputs improve learning speed.
- Useful in hidden layers.

### Cons:

- Still suffers from vanishing gradients for large magnitudes.
- Computationally more expensive than ReLU.

### Use case:

- Hidden layers in small to medium-sized networks.

---

## 6. ReLU (Rectified Linear Unit)

### Definition:

$$
f(x) = \max(0, x)
$$

### Explanation:

- Outputs zero for negative inputs, linear for positive.
- Introduces sparsity (many neurons inactive).

### Properties:

- Differentiable everywhere except at zero (works fine in practice).
- No vanishing gradient for positive inputs.

### Pros:

- Efficient computation.
- Mitigates vanishing gradient problem.
- Encourages sparse activations, often improving generalization.

### Cons:

- Can cause "dying ReLU" problem (neurons stuck outputting 0).
- Not zero-centered.

### Use case:

- Most common choice for hidden layers in deep networks.

---

## 7. Softmax Activation Function

### Definition:

For an input vector $\mathbf{z} = [z_1, z_2, ..., z_K]$ of length $K$, the softmax function outputs a vector $\mathbf{y} = [y_1, y_2, ..., y_K]$ where:

$$
y_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

### Explanation:

- Converts the input vector into a **probability distribution** over $K$ classes.
- Each output $y_i$ is between 0 and 1, and all outputs sum to 1.
- Useful for **multi-class classification** tasks.

### Properties:

- Differentiable everywhere.
- Outputs normalized probabilities.
- Amplifies differences between inputs (the highest input gets the highest output probability).

### Pros:

- Ideal for multi-class classification output layers.
- Allows model to predict the most likely class.
- Works well with cross-entropy loss for training.

### Cons:

- Computationally expensive for very large output spaces.
- Sensitive to very large or very small input values (can be stabilized numerically).

### Use case:

- Output layer for multi-class classification problems (e.g., recognizing digits 0–9).

### When to Use Softmax?

- When your output must represent a **probability distribution** over multiple classes.
- Use in the **final output layer** of a network performing multi-class classification.
- Commonly paired with **cross-entropy loss** for training.

---

## **When to Choose Which Activation?**

| Task/Layer                                        | Recommended Activation(s)          | Why?                                                           |
| ------------------------------------------------- | ---------------------------------- | -------------------------------------------------------------- |
| **Output layer (binary classification)**          | Sigmoid                            | Outputs probability between 0 and 1.                           |
| **Output layer (multi-class classification)**     | Softmax                            | Probability distribution over classes.                         |
| **Output layer (regression)**                     | Linear                             | Predict continuous values directly.                            |
| **Hidden layers (shallow networks)**              | tanh or sigmoid                    | Smooth gradients, zero-centered (tanh better).                 |
| **Hidden layers (deep networks)**                 | ReLU (or variants like Leaky ReLU) | Fast convergence, avoids vanishing gradients.                  |
| **Simple perceptron models**                      | Step (threshold) or sign           | Binary decisions for linearly separable problems.              |
| **Output layer (multi-class with probabilities)** | Softmax                            | Normalizes outputs to sum to 1, preserves relative magnitudes. |

---

# **How to Combine Activation Functions in a Network?**

- Use **non-linear activation functions** (ReLU, tanh, sigmoid) in **hidden layers** to introduce non-linearity.
- Use a **linear activation** in the output layer if the output is continuous (regression).
- Use **sigmoid** in output for binary classification problems.
- For deep networks, avoid sigmoid/tanh in hidden layers because of vanishing gradient; prefer **ReLU**.
- Sometimes you might combine **ReLU hidden layers** with a **sigmoid/tanh output layer**, depending on task.
- For multi-class classification, the output usually uses **softmax** (generalization of sigmoid), but that’s beyond perceptrons.

---

# **Complete List of Activation Functions**

| Activation Function  | Formula/Definition                                 | Output Range      | Differentiable? | Zero-centered? | Typical Use Case                            |
| -------------------- | -------------------------------------------------- | ----------------- | --------------- | -------------- | ------------------------------------------- |
| **Linear**           | $f(x) = x$                                         | $-\infty, \infty$ | Yes             | Yes            | Output layer for regression problems        |
| **Threshold (Step)** | $f(x) = 1 \text{ if } x \geq 0 \text{ else } 0$    | {0, 1}            | No              | No             | Classic perceptron binary classification    |
| **Sign**             | $f(x) = 1 \text{ if } x \geq 0 \text{ else } -1$   | {-1, 1}           | No              | Yes            | Binary classification with $\pm 1$ labels   |
| **Sigmoid**          | $\sigma(x) = \frac{1}{1 + e^{-x}}$                 | (0, 1)            | Yes             | No             | Binary classification output layers         |
| **tanh**             | $\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$ | (-1, 1)           | Yes             | Yes            | Hidden layers (small/medium networks)       |
| **ReLU**             | $f(x) = \max(0, x)$                                | \[0, $\infty$)    | Yes (almost)    | No             | Hidden layers in deep networks              |
| **Softmax**          | $y_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$             | \[0, 1], sum=1    | Yes             | No             | Output layer for multi-class classification |

---

# **Summary:**

- **Linear:** Regression output.
- **Step/Sign:** Classic binary classifiers, non-differentiable, simple perceptrons.
- **Sigmoid:** Output probabilities, smooth, but vanishing gradient.
- **tanh:** Zero-centered, better than sigmoid for hidden layers, still has vanishing gradient.
- **ReLU:** Default choice for hidden layers in deep nets, sparse activations, fast training.

---
