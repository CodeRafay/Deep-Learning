# **1. Parameter Norm Penalty**

### **Concept**

- Also called **weight regularization**.
- Adds a penalty term to the **loss function** that discourages large weight values.
- Idea: smaller weights → simpler models → less chance of overfitting.

### **Types**

1. **L2 Regularization (Ridge)**

   - Penalty:

     $$
     \lambda \sum_{i} w_i^2
     $$

   - Encourages weights to be small and spread out.
   - Prevents overfitting, improves stability, works well in deep learning.

2. **L1 Regularization (Lasso)**

   - Penalty:

     $$
     \lambda \sum_{i} |w_i|
     $$

   - Encourages sparsity (many weights become zero).
   - Useful for **feature selection**.

3. **Elastic Net**

   - Combination of L1 and L2 penalties.
   - Balances sparsity with weight shrinkage.

### **Effect on Convergence**

- Slows down growth of weights, forcing the optimizer to find smoother solutions.
- Works well with gradient-based optimizers.

---

# **2. Dataset Augmentation**

### **Concept**

- Artificially increasing the size and diversity of training data by applying transformations.
- Prevents overfitting by exposing the network to more variations of data.

### **Examples**

- **Computer Vision**:

  - Flipping, rotation, cropping, scaling, brightness/contrast changes, adding noise.
  - Cutout, Mixup, CutMix (advanced augmentations).

- **NLP**:

  - Synonym replacement, back-translation, random word insertion/deletion, paraphrasing.

- **Audio**:

  - Time-shifting, pitch shifting, background noise, speed changes.

### **Effect on Convergence**

- Slows training a bit (more diverse inputs), but improves **generalization**.
- Acts like data-driven regularization.

---

# **3. Dropout**

### **Concept**

- Randomly “drops” (sets to zero) a fraction of neurons during training.
- Prevents co-adaptation of neurons (when neurons rely too much on each other).

$$
h_i^{drop} = \begin{cases}
0 & \text{with probability } p \\
\frac{h_i}{1-p} & \text{otherwise}
\end{cases}
$$

where $p$ is the dropout rate (e.g., 0.5).

### **Why It Works**

- Each training step uses a slightly different network architecture.
- Equivalent to training an ensemble of smaller networks and averaging their predictions.

### **Effect on Convergence**

- Slows convergence slightly (since fewer neurons are active each step).
- Greatly improves **generalization**.
- At inference time, all neurons are used but scaled accordingly.

### **Variants**

- **Spatial Dropout**: Drops entire feature maps in CNNs.
- **DropConnect**: Randomly drops weights instead of activations.

---

# **4. Batch Normalization (BN) and Its Effect on Convergence**

### **What is Batch Normalization?**

- A technique introduced to **normalize activations** within a layer during training.
- Each mini-batch’s activations are normalized to have **zero mean** and **unit variance**, then scaled and shifted with learnable parameters ($\gamma, \beta$).

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

Where:

- $\mu_B$, $\sigma_B^2$ → mean and variance of mini-batch.
- $\epsilon$ → small constant for numerical stability.

---

### **Why BN Helps Convergence**

1. **Reduces Internal Covariate Shift**:

   - As parameters update, the distribution of activations changes.
   - BN stabilizes these distributions, making training smoother.

2. **Allows Higher Learning Rates**:

   - Without BN, high learning rates often cause divergence.
   - BN smooths the loss surface, so larger steps can be taken.

3. **Acts as Regularization**:

   - Adds small noise due to batch statistics.
   - Reduces overfitting, sometimes making **Dropout less necessary**.

4. **Improves Gradient Flow**:

   - Prevents gradients from vanishing or exploding, especially in deep networks.

5. **Speeds Up Convergence**:

   - Networks often converge in fewer epochs when BN is used.

---

### **Trade-offs of Batch Normalization** / **Limitations**

- Depends on **batch size** (unstable if batches are too small).
- Adds **computational overhead**.
- For **Recurrent Neural Networks (RNNs)**, BN is tricky due to sequence dependency → alternatives like **Layer Normalization** or **Group Normalization** are used.

---

# **Comparison and Interplay**

| Technique                          | Primary Goal            | How it Works              | Effect on Training         | Effect on Generalization           |
| ---------------------------------- | ----------------------- | ------------------------- | -------------------------- | ---------------------------------- |
| **Parameter Norm Penalty (L1/L2)** | Control weight growth   | Adds penalty term to loss | Slower but stable updates  | Reduces overfitting, sparsity (L1) |
| **Dataset Augmentation**           | Increase data diversity | Transformations of data   | Training longer but richer | Strong generalization              |
| **Dropout**                        | Prevent co-adaptation   | Random neuron removal     | Slower convergence         | Excellent generalization           |
| **Batch Normalization**            | Stabilize distributions | Normalize activations     | Faster convergence         | Mild regularization                |

---

✅ Together, these techniques often complement each other:

- Use **L2 regularization** + **BatchNorm** for stability.
- Add **Dropout** for generalization.
- Apply **Dataset Augmentation** if dataset is small.
