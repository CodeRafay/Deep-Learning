## Introduction to CNN Architectures

In the field of deep learning, **Convolutional Neural Networks (CNNs)** have revolutionized the way machines interpret and classify images. CNNs are specially designed to process data with a **grid-like topology**, such as images, by applying **convolution operations** to extract hierarchical features — from simple edges to complex patterns.

Over the years, various CNN architectures have been developed, each improving upon the previous in terms of depth, accuracy, and computational efficiency. Understanding these architectures is essential for:

- Analyzing model design choices,
- Comparing performance across tasks,
- Gaining intuition on how deep learning models interpret visual data.

This section explores **popular CNN architectures**, starting with the **LeNet-5 model**, which laid the foundation for modern deep learning approaches in computer vision.

---

# 1. [LeNet-5 – Classic CNN Architecture](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

### **Introduction**

- **Developed by**: Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner (1998)
- **Purpose**: Handwritten and machine-printed character recognition.
- **Dataset**: Designed for **MNIST** (digits 0–9).
- **Significance**: One of the **first Convolutional Neural Networks** (CNNs) – foundational model in deep learning history.

---

### **Architecture Overview**

- **Input**: Grayscale image of size **32×32×1**
- **Total Parameters**: ~**60,000**

---

### **Layer-wise Breakdown**

| Layer # | Type                                   | Details                                | Output Size |
| ------- | -------------------------------------- | -------------------------------------- | ----------- |
| 1       | **Input**                              | 32×32 grayscale image                  | 32×32×1     |
| 2       | **C1 – Convolution**                   | 6 filters, 5×5 kernel, stride=1        | 28×28×6     |
| 3       | **S2 – Average Pooling**               | 2×2 pool, stride=2                     | 14×14×6     |
| 4       | **C3 – Convolution**                   | 16 filters, 5×5 kernel                 | 10×10×16    |
| 5       | **S4 – Average Pooling**               | 2×2 pool, stride=2                     | 5×5×16      |
| 6       | **C5 – Convolution / FC**              | 120 filters, 5×5 kernel (acts like FC) | 1×1×120     |
| 7       | **F6 – Fully Connected**               | 84 neurons                             | 84          |
| 8       | **Output – Fully Connected + Softmax** | 10 classes (digits 0–9)                | 10          |

---

![LeNet-5 Architecture](https://miro.medium.com/v2/resize:fit:800/format:webp/1*DMcPgeekUftwk0GTMcNawg.png)

---

### **Key Concepts**

- **Convolution Layers**:

  - Extract **local spatial features**.
  - Use small filters (e.g., 5×5).
  - Apply activation functions (typically **tanh** or **sigmoid** in LeNet-5).

- **Average Pooling Layers**:

  - Downsample feature maps.
  - Reduces computation and helps generalization.

- **Fully Connected Layers**:

  - Learn high-level features.
  - Perform final classification using **Softmax**.

- **Activation Function**:

  - Originally used **tanh/sigmoid**; modern variants may use **ReLU**.

---

### **Key Concepts**

- **Why 32×32 input**: MNIST digits are 28×28, padded to 32×32 to allow better edge detection in convolutions.
- **Pooling type**: Used **Average Pooling**, not Max Pooling (which is more common today).
- **Depth increases**: From 1 channel to 6, 16, 120 as we go deeper.
- **Feature abstraction**: Shallow layers detect **edges**, deeper layers detect **shapes**, final layers detect **digit representations**.
- **Parameter efficiency**: Despite being deep, uses far fewer parameters than modern networks.

---

### **Summary**

- LeNet-5 is a **pioneering CNN** model.
- Designed for **simple image classification tasks**.
- Important for understanding **basic CNN components**.
- A good example of how **feature extraction and classification** are combined in a neural network.

---

# 2. [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

### **Introduction**

- **Developed by**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012)
- **Purpose**: Large-scale image classification on ImageNet dataset (1.2 million images, 1000 classes)
- **Significance**: Winner of the **ImageNet ILSVRC-2012** competition; pivotal model that revived deep learning research in computer vision
- **Training**: Used two Nvidia GeForce GTX 580 GPUs; network was split into two pipelines due to hardware limitations
- **Parameters**: Approximately **60 million**

---

### **Architecture Overview**

- **Layers**: 8 layers in total

  - 5 convolutional layers
  - 3 fully connected layers

- **Input Size**: 224×224×3 (RGB images)
- **Key Innovations**:

  - **ReLU activation** for faster training and better gradient flow
  - **Max Pooling** instead of Average Pooling for better feature downsampling
  - **Dropout** in fully connected layers to reduce overfitting
  - **Data Augmentation** to artificially increase dataset size and improve generalization
  - **Local Response Normalization (LRN)** introduced to mimic lateral inhibition observed in real neurons
  - **Stochastic Gradient Descent (SGD)** optimization

---

### **Layer-wise Breakdown**

| Layer # | Type            | Details                                  | Output Size |
| ------- | --------------- | ---------------------------------------- | ----------- |
| 1       | Convolution     | 96 filters, 11×11 kernel, stride=4, ReLU | 55×55×96    |
| 2       | Max Pooling     | 3×3 pool, stride=2                       | 27×27×96    |
| 3       | Convolution     | 256 filters, 5×5 kernel, padding=2, ReLU | 27×27×256   |
| 4       | Max Pooling     | 3×3 pool, stride=2                       | 13×13×256   |
| 5       | Convolution     | 384 filters, 3×3 kernel, padding=1, ReLU | 13×13×384   |
| 6       | Convolution     | 384 filters, 3×3 kernel, padding=1, ReLU | 13×13×384   |
| 7       | Convolution     | 256 filters, 3×3 kernel, padding=1, ReLU | 13×13×256   |
| 8       | Max Pooling     | 3×3 pool, stride=2                       | 6×6×256     |
| 9       | Fully Connected | 4096 neurons, ReLU                       | 4096        |
| 10      | Fully Connected | 4096 neurons, ReLU                       | 4096        |
| 11      | Fully Connected | 1000 neurons (output classes), Softmax   | 1000        |

---

### **Key Concepts**

- **ReLU Activation**

  - Replaced sigmoid/tanh to reduce vanishing gradient problem and accelerate training

- **Max Pooling**

  - More effective than average pooling in highlighting dominant features

- **Dropout**

  - Randomly disables neurons during training to prevent overfitting

- **Data Augmentation**

  - Techniques like image flipping, cropping, and color jittering increase training data diversity

- **Local Response Normalization (LRN)**

  - Encourages competition among neurons, improving generalization

- **Use of GPUs**

  - Allowed training of deeper and larger models, crucial for practical deep learning

---

### **Key Concepts**

- AlexNet marked the transition from shallow to **deep CNN architectures** for large-scale vision tasks.
- Introduced several key innovations that are standard in modern CNNs: **ReLU, dropout, data augmentation, and LRN**.
- Utilized **multiple GPUs** due to computational requirements, which was a significant technical advancement at the time.
- Despite high performance, it had a large number of **hyperparameters** making tuning complex.
- Model size (~60 million parameters) is significantly larger than LeNet-5 (~60,000 parameters).

---

### **Summary**

- AlexNet is a landmark CNN architecture that demonstrated the power of deep learning on a large-scale dataset.
- Its success reignited research interest in CNNs and deep learning for computer vision.
- Innovations such as ReLU, dropout, and GPU-based training are now foundational in CNN design.

---

## ![AlexNet Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*i7uy7w20u7tuMmwUNohKEw.png)

# 3. [VGG-16 Net](https://arxiv.org/pdf/1409.1556)

### **Introduction**

- **Developed by**: Karen Simonyan and Andrew Zisserman (2014)
- **Purpose**: Improve on AlexNet by reducing the number of hyperparameters and increasing network depth with a more uniform architecture
- **Achievement**: 1st runner-up in the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014**
- **Parameters**: Approximately **138 million**

---

### **Architecture Overview**

- **Input**: RGB image of size **224×224×3**
- **Depth**: 16 weight layers (13 convolutional + 3 fully connected)
- **Key Design Change**:

  - Replaced large convolutional kernels (e.g., 11×11, 5×5 in AlexNet) with multiple stacked **3×3 convolutional filters**
  - Used **2×2 max pooling** with stride 2 (not stride 1) to downsample while preserving important features
  - Padding applied to keep spatial dimensions consistent after convolution

---

### **Layer-wise Breakdown**

| Block | Layer Type      | Filters / Neurons | Repetitions | Output Size    |
| ----- | --------------- | ----------------- | ----------- | -------------- |
| 1     | Convolution     | 64 filters (3×3)  | 2           | 224×224×64     |
|       | Max Pooling     | 2×2, stride=2     | 1           | 112×112×64     |
| 2     | Convolution     | 128 filters (3×3) | 2           | 112×112×128    |
|       | Max Pooling     | 2×2, stride=2     | 1           | 56×56×128      |
| 3     | Convolution     | 256 filters (3×3) | 3           | 56×56×256      |
|       | Max Pooling     | 2×2, stride=2     | 1           | 28×28×256      |
| 4     | Convolution     | 512 filters (3×3) | 3           | 28×28×512      |
|       | Max Pooling     | 2×2, stride=2     | 1           | 14×14×512      |
| 5     | Convolution     | 512 filters (3×3) | 3           | 14×14×512      |
|       | Max Pooling     | 2×2, stride=2     | 1           | 7×7×512        |
|       | Fully Connected | 4096 neurons      | 2           | 4096           |
|       | Fully Connected | 1000 neurons      | 1           | 1000 (classes) |

---

### **Key Concepts**

- **Use of small 3×3 kernels** stacked consecutively:

  - Equivalent to a larger receptive field (e.g., two 3×3 conv layers have effective 5×5 receptive field)
  - Reduces number of parameters and computational cost compared to large kernels

- **Consistent use of padding** to maintain spatial resolution after convolutions
- **Max Pooling** layers reduce spatial dimension by half each time
- **Fully connected layers** at the end perform classification
- **ReLU activation** used throughout the network for non-linearity

---

### **Key Concepts**

- VGG-16 solved the problem of too many hyperparameters in AlexNet by using a **uniform architecture** with smaller convolution kernels.
- The network is **very deep (16 layers)** compared to AlexNet (8 layers), which improved learning capacity and accuracy.
- Though powerful, VGG-16 is **computationally expensive**, with **high memory requirements** and **longer training times**.
- Suffers from **vanishing/exploding gradient problems** due to depth, making training more difficult without proper techniques (later addressed by ResNet).
- The large number of parameters (~138 million) makes it a heavy model for practical applications without hardware acceleration.

---

### **Summary**

- VGG-16 is a deep CNN architecture with a simple and uniform design based on stacked 3×3 convolutional filters.
- It improved classification accuracy significantly by increasing depth while controlling parameters.
- Its **drawbacks** include heavy computational requirements and difficulty in training, motivating the development of more efficient models later.

---

## ![VGG-16 Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*B_ZaaaBg2njhp8SThjCufA.png)

# 4. [ResNet](https://arxiv.org/pdf/1512.03385)

### **Introduction**

- **Developed by**: Kaiming He et al. (2015)
- **Purpose**: Address the degradation problem in very deep neural networks by enabling effective training of networks with over 100 layers
- **Achievement**: Winner of **ILSVRC 2015** competition
- **Key Innovation**: Introduction of **skip connections (residual connections)** and **batch normalization**

---

### **Architecture Overview**

- ResNet builds on architectures like VGG by stacking many layers but introduces **residual blocks** with **identity skip connections**
- These skip connections allow the network to learn a residual mapping instead of directly fitting the desired underlying mapping
- Enables training of **very deep networks** (e.g., ResNet-50, ResNet-101, ResNet-152) without degradation in accuracy
- Uses **Batch Normalization** to stabilize and speed up training

---

## ![Normal Deep Networks vs Networks with skip connections](https://miro.medium.com/v2/resize:fit:720/format:webp/1*2gQ6vueMAv-PeKNFqyNDVw.png)

### **Core Concepts**

- **Skip Connections / Residual Learning**

  - Allows gradients to flow directly through the network by bypassing one or more layers
  - If the weights of a layer degrade to zero, the output can still pass through unchanged (identity mapping).

- **Formula for residual block output**:

  ```
  a[l+2] = g(w[l+2] * a[l+1] + a[l])
  ```

  Where:

  - `a[l]` is the input to the residual block
  - `w[l+2]` represents the weights of the layer
  - `g` is the activation function

- **Vanishing Gradient Problem**

  - In very deep networks, gradients can become very small, preventing effective learning
  - Skip connections mitigate this by providing alternate paths for gradient flow

- **Batch Normalization**

  - Normalizes layer inputs to reduce internal covariate shift, improving training speed and stability

---

### **Significance**

- Before ResNet, increasing network depth beyond a certain point caused accuracy to saturate or degrade
- ResNet’s skip connections enable training of networks with hundreds or even thousands of layers, significantly improving performance on image recognition tasks
- Inspired by earlier ideas like **highway networks** with gated shortcuts and similar to skip connections in **LSTMs** for sequential data

---

### **Summary**

- ResNet introduced a fundamental architectural change with **residual connections** allowing very deep CNNs to be trained effectively
- It overcame limitations of previous deep networks caused by vanishing gradients and degradation of performance
- ResNet models remain a strong baseline for modern deep learning research and applications

---

# 5. [Inception Net (GoogLeNet)](https://arxiv.org/pdf/1409.4842)

### **Introduction**

- **Proposed by**: Researchers at Google in the paper _“Going Deeper with Convolutions”_ (2014)
- **Alternative Name**: **GoogLeNet**
- **Purpose**: Efficiently capture both **local** and **global features** by applying multiple convolution kernel sizes in parallel
- **Key Motivation**:

  - In real-world images, **salient features vary greatly in size**
  - Choosing the right kernel size becomes difficult
  - Inception module solves this by applying **multiple kernel sizes (1×1, 3×3, 5×5)** in parallel

- **Model Depth**:

  - **22 layers deep** (27 if pooling layers are counted)
  - Contains **9 Inception modules**

---

### **Core Idea – Inception Module**

The **Inception module** applies **multiple filters in parallel** and then **concatenates the results** along the depth (channel) dimension.

![Inception Module of GoogleLeNet](https://miro.medium.com/v2/resize:fit:720/format:webp/1*vqwdhwFnVNiT2XTKe2DrWw.png)

**Note**: Same padding is used to preserve the dimension of the image.

**How it works**:

- Applies 1×1, 3×3, and 5×5 convolutions **simultaneously**
- Applies **Max Pooling** in parallel as well
- Concatenates all outputs and passes to the next layer
- This allows the network to capture features at **multiple receptive fields** in the same layer

---

### **Parameter Reduction – Bottleneck Technique**

- Direct use of 3×3 and 5×5 kernels increases the number of parameters significantly (initial model ~120M parameters)
- To address this, **1×1 convolutions** are used **before** applying larger convolutions

  - Act as **bottlenecks** that reduce depth (number of channels)
  - Dramatically reduce computational cost and parameters

- With this trick, total parameters reduced by **~90%**

---

### **Network Structure**

GoogLeNet stacks multiple inception modules to form a **deep and wide** architecture.

![Several Inception modules are linked to form a dense network](https://miro.medium.com/v2/resize:fit:720/format:webp/1*BSDA-IImTx9NIHpoHa2E0w.png)

**What this image shows**:

- Multiple inception modules are linked together
- **Side branches** (auxiliary classifiers) predict outputs at intermediate depths

  - Help with vanishing gradients
  - Provide regularization and better convergence during training

---

### **Key Concepts**

- **Multi-scale feature extraction** within the same layer
- **1×1 convolutions** used for both dimensionality reduction and as activation functions
- **Parallel structure** rather than sequential stacking
- **Global average pooling** replaces fully connected layers
- **Auxiliary classifiers** help with training deeper models

---

### **General Observations**

- Inception Net provided a significant improvement in both **accuracy** and **efficiency**
- Architecture is **modular and scalable**, allowing for variants like **Inception v2**, **v3**, and **v4**
- The model structure is **wider** (parallel operations), not just deeper
- Bottleneck layers using 1×1 convolutions are crucial in managing computational cost

---

### **Summary**

- Inception Net (GoogLeNet) introduced a new approach of **multi-kernel convolution within the same layer**
- Efficient in handling varying spatial features while controlling parameter growth
- The model’s modular design inspired many **modern deep learning architectures** focused on both **depth and width**
- Later versions (v2, v3, v4) further optimized training and reduced computational complexity
