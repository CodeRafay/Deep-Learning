# **Convolutional Neural Networks (CNNs)**

Convolutional Neural Networks are a specialized class of deep neural networks designed to **process and understand spatial data**, particularly **images and videos**. They exploit the **spatial structure of pixels** — meaning the relationship between neighboring pixels — which standard Artificial Neural Networks (ANNs) fail to capture when data is flattened into one-dimensional vectors.

---

## **1. Neural Networks for Spatial Data**

### **a) Nature of spatial data**

Unlike textual or tabular data, **image data** has an inherent **spatial structure**: pixels close to each other are often **correlated** and collectively represent patterns like edges, corners, or textures.

Spatial data refers to data that contains **positional or locational relationships** between elements — such as images, videos, or even 3D medical scans.refers to **how pixels relate to one another**:

- Their **positions** (x, y in an image).
- Their **relationships** (edges, corners, textures).
- The **patterns** formed by nearby pixel intensities.

For example:

- In facial recognition, spatial info defines the _relative position of eyes, nose, and mouth_.
- In object detection, it defines _edges, boundaries, and depth cues_.

CNNs preserve this **spatial structure** by learning **filters** that detect spatial patterns like edges, curves, and shapes directly from image data.

In this century, such spatial data dominates real-world applications:

- Smart city systems (e.g., vehicle detection in surveillance)
- Autonomous vehicles (object recognition, road segmentation)
- Facial recognition systems
- Medical imaging (CT/MRI scans)
- Satellite image analysis

### **b) How ANN processes images**

A standard **ANN** works well for **tabular or textual data** because it expects each input feature to be independent. In other words, it expects **1D vector inputs**. However, **images are different**:

- An image is **2D (or 3D for color)**, with strong **spatial correlations** between nearby pixels.
- Neighboring pixels often represent connected edges, textures, or objects.

Flattening such data into a **1D vector** (as ANN requires) destroys this relationship.
So, an image with shape, say, **(32 × 32 × 3)** (height, width, RGB channels), must be **flattened** into a **1D vector of size 3072 (32×32×3)** before being fed into the network.

Example:

```
Original Image: 32×32×3 = 3072 values
Flattened Vector: [x₁, x₂, x₃, ..., x₃₀₇₂]
```

Consider another simple **3×3 grayscale image**:

[
\begin{bmatrix}
10 & 20 & 30 \
40 & 50 & 60 \
70 & 80 & 90
\end{bmatrix}
]

Flattening this image for an ANN gives:
[
[10, 20, 30, 40, 50, 60, 70, 80, 90]
]

Now, pixels that were **spatially adjacent** (e.g., 20 and 50, vertically neighbors) are **far apart** in the flattened representation.
The **spatial information** — the geometry, edges, and textures is lost. Flattening removes the **spatial structure** — the positional relationships among neighboring pixels. For instance, the top-left corner pixel and bottom-right corner pixel become neighbors in the flattened vector, which **destroys spatial locality**.

This is why **ANNs are poor at image tasks** like classification, segmentation, or object detection. They don’t understand spatial locality, shapes, or visual hierarchy.

This is problematic because:

- In images, **local patterns** like edges, textures, or shapes carry meaning.
- ANNs treat all inputs as **independent features**, so they cannot leverage the **2D neighborhood relationships** between pixels.

### **c) Why ANNs fail on spatial data**

ANNs require massive numbers of parameters when dealing with high-dimensional images (for example, an image of 224×224×3 has 150,528 pixels). Connecting each pixel to every neuron in the first hidden layer would mean **millions of parameters**, making the model **slow, memory-intensive, and prone to overfitting**.
Moreover, **ANNs are position-agnostic** — they do not understand that the same object can appear in different parts of an image.

This is where **Convolutional Neural Networks (CNNs)** revolutionized the field.

---

## **2. History and Evolution of CNNs**

### **a) Origins**

CNNs were inspired by the **biological visual cortex** — specifically, how neurons in the visual cortex respond to overlapping local receptive fields in the visual input.
Yann LeCun’s **LeNet-5 (1989)** was among the first successful CNNs, used for **handwritten digit recognition** (the famous MNIST dataset).

Later milestones:

- **AlexNet (2012)**: Won the ImageNet competition, using ReLU activations, dropout, and GPUs for faster training.
- **VGGNet (2014)**: Demonstrated the power of depth (very deep networks).
- **GoogLeNet/Inception (2014)**: Used multi-scale convolutional filters.
- **ResNet (2015)**: Introduced **skip connections**, enabling very deep models.
- **Vision Transformers (ViTs, 2020)**: Combined self-attention with image patches.
- **VLMs (Vision-Language Models)**: CLIP, DINOv2, and others connect visual and textual understanding.

---

### **b) Core Principle**

Unlike ANNs that process the entire input as a flat vector, CNNs process **small regions (local receptive fields)** at a time.
A **filter or kernel** slides over the image, performing **convolution** — a dot product between the filter and a local patch of the image.

This captures **local patterns** like edges, corners, and textures while maintaining the **spatial hierarchy** of features.
Each filter learns to detect a specific feature: vertical edges, horizontal lines, color gradients, or object parts.

---

### **c) Key Components of a CNN**

1. **Convolutional Layers:**
   Perform convolution operations to extract features.
   Each convolutional layer applies multiple filters, producing feature maps.

   - Early layers: detect simple edges, colors, or orientations.
   - Deeper layers: detect complex structures like eyes, faces, cars, or textures.

   Mathematically:
   [
   (I * K)(x, y) = \sum_{i}\sum_{j} I(x+i, y+j)K(i, j)
   ]
   where ( I ) = input image, ( K ) = kernel/filter.

2. **Activation Functions:**
   Typically **ReLU (Rectified Linear Unit)** is used, as it introduces **non-linearity** and avoids vanishing gradients:
   [
   f(x) = \max(0, x)
   ]
   ReLU helps CNNs converge faster compared to sigmoid or tanh.

3. **Pooling Layers (Subsampling):**
   Reduce the spatial dimensions while retaining key features.
   Types:

   - **Max Pooling:** Keeps the maximum value in a patch (commonly used)
   - **Average Pooling:** Takes the mean of the patch
     Pooling adds **spatial invariance** — the model becomes robust to small translations or distortions.

4. **Fully Connected (FC) Layers:**
   After several convolutions and poolings, the output feature maps are **flattened** and passed through dense layers for classification.
   These layers act as the **decision-making part** of the network.

5. **Output Layer:**

   - For **classification**, Softmax activation is used to output class probabilities.
   - For **detection/segmentation**, bounding boxes or masks are generated instead.

---

### **d) Spatial Feature Preservation**

Unlike ANNs, CNNs maintain **spatial relationships** by using local receptive fields and shared weights.
This means neighboring pixels remain neighbors during computation, allowing the network to recognize **patterns in context**.

For instance:

- The edges of a car door and window are spatially related.
- Flattening this (like in ANN) would destroy that relationship.

By using filters sliding across 2D input, CNNs **retain spatial structure** and **reuse weights**, leading to **translation invariance** and **parameter efficiency**.

In CNNs:

1. **Convolution + Pooling layers** act as **feature extractors**.
2. **Fully Connected layers** act as **classifiers**.
   This division makes CNNs efficient and modular.

---

### **e) Feature Extraction in CNNs vs Traditional Methods**

Before CNNs, computer vision used **handcrafted feature extractors** like:

- **HOG (Histogram of Oriented Gradients)** – captures edge directions.
- **LBP (Local Binary Patterns)** – encodes texture by comparing pixel intensities.
- **SIFT, SURF, BoW (Bag of Words)** – detect and describe keypoints.

These required **manual design and domain expertise**, and they were **not trainable**.
CNNs replaced these by **learning hierarchical features directly from data**, automatically discovering what matters for the task.

---

### **f) Applications of CNNs**

CNNs dominate all visual tasks:

- **Classification:** ImageNet, CIFAR, MNIST
- **Detection:** YOLO, Faster R-CNN
- **Segmentation:** U-Net, Mask R-CNN
- **Reconstruction:** Autoencoders, Super-Resolution
- **Vision-Language:** CLIP, DINOv2, SAM, GPT-4V

Thus, CNNs are the backbone of **modern computer vision** and remain unbeaten for image-based deep learning.

---

### **g) Feature Hierarchy**

CNNs build hierarchical understanding:

- **Initial layers:** General low-level patterns (edges, textures, colors)
- **Middle layers:** Intermediate shapes or object parts
- **Deep layers:** High-level semantic features (faces, cars, digits)

This **progressive abstraction** makes CNNs powerful and interpretable in stages.

---

### **h) Visual-Language Models (VLMs)**

Recently, CNNs have been fused with **language models** to create **multimodal AI** — systems that understand both **text and images**.
Examples:

- **CLIP (Contrastive Language-Image Pretraining)**: Aligns visual and textual features in a shared space.
- **BLIP, Flamingo, DINOv2, SAM (Segment Anything Model)**: Push the boundary of visual understanding.
  VLMs often use **CNN backbones or vision transformers** to extract embeddings.

---

### **i) CNN Summary Pipeline**

The overall CNN flow:

```
Input Image
   ↓
Convolution Layer (Feature Extraction)
   ↓
Activation (ReLU)
   ↓
Pooling (Subsampling)
   ↓
Convolution + Pooling (deeper patterns)
   ↓
Flatten
   ↓
Fully Connected Layer (Decision)
   ↓
Softmax / Output Layer
```

This architecture enables the network to **preserve spatial structure**, **reduce parameters**, and **learn automatically** from raw image data.

---

### **j) Why CNNs are Superior**

- Handle high-dimensional image data efficiently
- Maintain spatial relationships
- Learn hierarchical feature representations
- Have translation and deformation invariance
- Far outperform traditional ANNs and handcrafted features

In short:

> For images, **no model beats CNNs** in combining accuracy, efficiency, and interpretability. Even with the rise of transformers, CNNs remain fundamental in deep learning.

---

## **3. CNNs – Convolutional Layers**

### **1. What is a Convolutional Layer?**

The **convolutional layer** is the **heart of a CNN**. It’s where the network performs the **feature extraction** — scanning the image using **small filters (kernels)** to detect visual patterns such as edges, corners, textures, and shapes.

When an image passes through a convolutional layer:

- A **small filter (F × F)** slides (or convolves) over the image.
- At each position, the filter and the local patch of the image are **multiplied element-wise and summed** to produce a single number.
- This single number becomes one element in the **feature map** (also called the **activation map**).

Mathematically:
[
O(x, y) = \sum_{i=0}^{F-1}\sum_{j=0}^{F-1} I(x+i, y+j) \times K(i, j)
]
where
(I) = input image, (K) = kernel, (O) = output feature map, (F) = kernel size.

---

### **2. How Convolution Works (Step-by-Step)**

Let’s take an example:

- Input image size: **6 × 6 × 1**
- Kernel size: **3 × 3**
- Stride: **1**
- Padding: **0**

**Step 1:**
The 3×3 kernel is placed on the top-left corner of the image.

**Step 2:**
Perform element-wise multiplication and sum all values to produce one pixel of the output feature map.

**Step 3:**
Move the kernel one step (based on stride) to the right, repeat until the entire image is covered.

**Result:**
If the image is (N × N) and kernel is (F × F) with **no padding and stride = 1**,
the output feature map size becomes:
[
(N - F + 1) × (N - F + 1)
]

---

### **3. Why Convolution is Powerful**

- **Local connectivity:** Each neuron is connected only to a small region of the input (local receptive field), not the entire input.
- **Parameter sharing:** The same filter (weights) is used across all image locations, drastically reducing the number of parameters.
- **Translation invariance:** The same feature can be detected anywhere in the image.

---

### **4. Hyperparameters in Convolutional Layers**

Hyperparameters are crucial to control how a CNN extracts features and how large or small its outputs become.

#### **(a) Kernel Size (F)**

- Determines **how large the region** the filter looks at.
- Typical sizes: 3×3, 5×5, or 7×7.
- Smaller kernels (3×3) capture fine details and reduce computation.
- Larger kernels capture broader patterns but require more memory and computation.

#### **(b) Number of Filters**

- Determines **how many features** are extracted.
- Each filter learns a different feature (edges, textures, curves, colors).
- Output depth (number of channels) equals the number of filters.

Example:

- Input: 32×32×3
- 10 filters of size 5×5
  → Output: 28×28×10 (without padding)

#### **(c) Stride (S)**

- Defines **how many pixels** the filter moves at each step.
- Stride = 1 → detailed scanning, larger output.
- Stride = 2 or more → faster scanning, smaller output (less information).

Formula (no padding):
[
\text{Output size} = \frac{(N - F)}{S} + 1
]

A higher stride **reduces output size** and **loses fine details**, but speeds up computation.

#### **(d) Padding (P)**

- Adds a border of zeros around the input image to **preserve spatial size** after convolution.
- Without padding, feature maps shrink after each convolution (since edges aren’t fully convolved).

**Two common types:**

- **Valid Padding:** No padding applied → output shrinks.
  Formula: ((N - F + 1))
- **Same Padding:** Pads image so that output size = input size.
  Formula with padding and stride:
  [
  \text{Output size} = \frac{(N - F + 2P)}{S} + 1
  ]

Padding helps retain border information but **increases computation and memory**.

---

### **5. Example Calculation**

Input: (32×32×3), Filter: (5×5×3×10), Stride = 1, Padding = 0
Output:
[
(32 - 5 + 1) × (32 - 5 + 1) × 10 = 28×28×10
]
→ 10 feature maps, each 28×28.

If Padding = 2:
[
(32 - 5 + 2×2)/1 + 1 = 32×32×10
]

---

### **6. The Role of Filters**

Each filter detects a specific **pattern or feature**:

- One filter might detect vertical edges,
- Another detects diagonal lines,
- Others might detect textures or object parts.

The **depth** of the CNN (number of layers) allows it to build up from **simple features to complex ones**.

---

### **7. From Convolution to Fully Connected Layers**

CNNs typically follow this pattern:

1. **Input Image**
2. **Convolutional Layers:** Extract features.
3. **Activation (ReLU):** Introduce non-linearity.
4. **Pooling Layers:** Reduce size while retaining key info.
5. **Flatten Layer:** Converts 3D feature maps into 1D vector.
6. **Fully Connected (Dense) Layers:** Combine features and make predictions.
7. **Output Layer:** Softmax or Sigmoid for classification.

---

## **4. CNN Architecture Overview**

The general architecture of a CNN is:

```
Input Image → Convolution → Activation (ReLU)
             ↓
          Pooling
             ↓
      Convolution + ReLU
             ↓
          Pooling
             ↓
          Flatten
             ↓
     Fully Connected Layers
             ↓
           Output
```

### **Key Notes:**

- **Convolution Layers:** Extract features.
- **Pooling Layers:** Downsample and retain key info.
- **ReLU or tanh:** Adds non-linearity.
- **Flatten Layer:** Converts 3D tensor to 1D vector.
- **Dense Layers:** Perform classification or regression.

---

## **5. CNNs – Pooling Layers**

Pooling is used to **reduce the spatial dimensions** of feature maps while **preserving important information**. It helps make CNNs **faster, less prone to overfitting**, and **translation-invariant**.

### **a) Purpose**

- To downscale the feature map and reduce parameters.
- To retain the most relevant features (e.g., strongest activations).
- To make the network less sensitive to small distortions or shifts.

### **b) Types of Pooling**

1. **Max Pooling**

   - Takes the **maximum value** in each region.
   - Keeps the strongest feature, making it robust to noise.
   - Most commonly used pooling type.

   Example:
   2×2 max pooling on an 8×8 image → output = 4×4.

2. **Average Pooling**

   - Takes the **mean** value of the region.
   - Retains smoother, more generalized features.
   - Often used in feature aggregation or global pooling layers.

3. **Min Pooling**

   - Keeps the minimum value in each patch.
   - Rarely used but can help when detecting dark/negative features.

4. **Global Average Pooling (GAP)**

   - Averages the entire feature map into one value per channel.
   - Commonly used before the final classification layer (e.g., ResNet).

---

### **c) Pooling and Output Size**

If input image = 8×8,
and pooling window = 2×2, stride = 2 → output = 4×4.
If window = 3×3, stride = 3 → output = 2×2.
So the image size reduces roughly by a factor of the pooling window.

Pooling reduces computation and the number of parameters, but also **reduces spatial precision**.

---

### **d) Learnable Parameters**

Pooling layers **have no learnable parameters**.
They perform fixed mathematical operations (max or average), unlike convolution layers that learn weights.
They simply summarize regions of the image.

---

### **e) Why Pooling is Important**

- Makes CNN **faster and lighter**.
- Adds **translation invariance** — a feature detected slightly to the left or right still triggers the same pooled result.
- Prevents **overfitting** by reducing feature dimensions.

---

### **f) Summary of CNN Processing Flow**

1. **Input Image:** 2D array (RGB: 3 channels)
2. **Convolutional Layer:** Filters slide to extract features.
3. **Activation (ReLU):** Adds non-linearity.
4. **Pooling:** Downsamples feature maps.
5. **Repeat Steps 2–4** for deeper abstraction.
6. **Flatten:** Convert feature maps to 1D vector.
7. **Fully Connected Layer:** Learn combination of features.
8. **Output:** Classification probabilities via Softmax.

---

## **6. Advanced CNN Operations**

CNNs might look simple — filters sliding across an image — but the actual mechanics involve numerous **design decisions** that determine accuracy, computational cost, and the model’s ability to generalize.
Let’s break down every part that influences this behavior.

---

### **1. Padding: Controlling Image Size and Edge Information**

When a convolution filter is applied to an image, pixels near the **edges** are used fewer times than pixels near the **center**.
Without handling this properly, the output size keeps shrinking after every layer, and **information near the borders is lost**.

Padding helps solve this by **adding extra pixels (often zeros)** around the border of the image before convolution.

#### **a) Valid Padding (No Padding)**

- No extra pixels are added.
- Output size becomes smaller after convolution.
- Formula:
  [
  \text{Output size} = (N - F + 1)
  ]
- Example: 6×6 input with 3×3 filter → Output = 4×4.

✅ Faster and less memory usage.
❌ Loses border information.
❌ Shrinks feature maps too quickly in deep networks.

---

#### **b) Same Padding**

- Adds padding ( P ) so that output size = input size.
- Formula:
  [
  \text{Output size} = \frac{N - F + 2P}{S} + 1
  ]
- Typically, ( P = \frac{F - 1}{2} ) when stride = 1.

✅ Keeps spatial dimensions constant.
✅ Preserves edge features.
❌ Slightly increases computational cost and training time.

---

### **2. Stride: Step Size of the Filter**

Stride determines **how far the kernel moves** across the image each time.

- Stride = 1 → Maximum overlap between patches → more detailed features.
- Stride = 2 → Skips alternate pixels → smaller output, less detail.

Formula (with padding):
[
\text{Output size} = \frac{N - F + 2P}{S} + 1
]

Example:

- (N=32), (F=3), (P=1), (S=2)
  [
  \text{Output} = \frac{32 - 3 + 2×1}{2} + 1 = 16
  ]
  So, 32×32 becomes 16×16.

✅ Higher stride = faster computation, fewer features.
❌ Too high = loss of critical details (less accurate).

---

### **3. Receptive Field: What Each Neuron Sees**

The **receptive field** is the region of the input image that a particular neuron in a given layer “sees” or is influenced by.

- In the **first layer**, each neuron sees a small patch (e.g., 3×3).
- As we move deeper, each neuron indirectly sees a **larger portion** of the original image due to stacking of multiple layers.

Example:

- Layer 1: 3×3 receptive field
- Layer 2: Each 3×3 filter on top of Layer 1 corresponds to 5×5 region of the input image.
- Deeper → exponentially larger receptive field.

Large receptive fields allow the network to capture **global context**, not just local edges.

---

### **4. Depth and Channel Operations**

CNNs handle multi-channel inputs naturally:

- A color image (RGB) has **3 channels**.
- A convolutional filter for RGB has dimensions **F × F × 3**.

Each filter produces one **feature map**.
If there are 64 filters, the output depth = 64.
Each filter learns different types of features (color gradients, edges, orientations, etc.).

---

### **5. Depthwise and Separable Convolutions**

To improve efficiency (especially in mobile or embedded devices), **depthwise separable convolutions** are used.

#### **a) Depthwise Convolution**

- Instead of convolving across all input channels at once, each channel is convolved separately with its own filter.
- Reduces computation drastically.

#### **b) Pointwise Convolution (1×1 Convolution)**

- A 1×1 convolution then combines the results from all channels to form new feature maps.
- Acts as a “feature mixer” across channels.

✅ Much fewer parameters.
✅ Used in **MobileNet**, **EfficientNet**, and **Xception**.
❌ Slightly less powerful than full convolutions if overused.

---

### **6. Dilated (Atrous) Convolutions**

Instead of using larger kernels, **dilated convolutions** insert **spaces between kernel elements**, expanding the receptive field without increasing parameter count.

Example:

- 3×3 filter with dilation rate = 2 → effectively covers a 5×5 area but only uses 9 parameters.

Used in:

- Semantic segmentation (e.g., DeepLab)
- Audio or time-series CNNs (for context extension)

✅ Captures long-range dependencies.
❌ Can cause “gridding” artifacts if not combined properly.

---

### **7. Transposed (Deconvolution) Layers**

Used in **decoder networks** or **autoencoders** for **upsampling** — increasing the spatial size of feature maps (the reverse of convolution).

Applications:

- Image generation (GANs)
- Image segmentation (U-Net, Decoder networks)

They learn how to **reconstruct fine-grained spatial details** from compressed feature representations.

---

### **8. Activation Functions in CNNs**

Every convolution layer is followed by an activation function that introduces **non-linearity**.

- **ReLU (Rectified Linear Unit)** → ( f(x) = \max(0, x) )
  → Most widely used; prevents vanishing gradients.
- **Leaky ReLU** → allows small gradient for negative inputs.
- **ELU (Exponential Linear Unit)** and **GELU** → used in advanced architectures for smoother activation.

Without activation, CNNs would act as **linear filters**, unable to capture complex relationships.

---

### **9. Modern CNN Architectures**

CNN design has evolved from simple stacks of convolutional layers to highly modular, efficient architectures:

#### **a) LeNet-5 (1989)**

- First successful CNN.
- Used for digit recognition.
- Architecture: Conv → Pool → Conv → FC → Output.

#### **b) AlexNet (2012)**

- Brought CNNs back to fame with ImageNet victory.
- Introduced **ReLU**, **Dropout**, **GPU training**.
- 5 conv layers, 3 FC layers.

#### **c) VGGNet (2014)**

- Used **3×3 convolutions** stacked deeply (up to 19 layers).
- Very uniform and simple design.
- Large number of parameters.

#### **d) GoogLeNet / Inception (2014)**

- Introduced **Inception modules**: parallel convolutions with different filter sizes (1×1, 3×3, 5×5) to capture features at multiple scales.
- Used **Global Average Pooling** instead of large FC layers.

#### **e) ResNet (2015)**

- Solved **vanishing gradient problem** with **skip (residual) connections**:
  [
  y = F(x) + x
  ]
- Allowed very deep networks (50, 101, 152 layers).
- Backbone for most modern vision systems.

#### **f) DenseNet (2016)**

- Each layer receives input from **all previous layers** (dense connectivity).
- Improves feature reuse and gradient flow.

#### **g) MobileNet (2017)**

- Uses **depthwise separable convolutions** to minimize parameters.
- Perfect for mobile/edge devices.

#### **h) EfficientNet (2019)**

- Scales network width, depth, and resolution **efficiently** using a single compound coefficient.

---

### **10. Receptive Field Growth Example**

| Layer | Kernel | Stride | Receptive Field Size |
| ----- | ------ | ------ | -------------------- |
| 1     | 3×3    | 1      | 3×3                  |
| 2     | 3×3    | 1      | 5×5                  |
| 3     | 3×3    | 1      | 7×7                  |
| 4     | 3×3    | 2      | 11×11                |

This growth shows that deeper layers have access to **larger portions of the image**, enabling **context-aware decisions**.

---

### **11. Parameter Sharing and Sparse Connections**

- **Parameter sharing:** The same filter is applied across all spatial locations, drastically reducing parameters compared to fully connected layers.
- **Sparse connectivity:** Each neuron connects to only a small subset of the previous layer (its receptive field), making CNNs more efficient and less redundant.

Example:
For a 5×5 filter and 3 input channels:
[
\text{Parameters per filter} = 5×5×3 = 75
]
Compare that to thousands in a dense layer — CNNs are far more efficient.

---

### **12. Summary of Advanced CNN Mechanics**

| Concept                  | Purpose                       | Effect                                       |
| ------------------------ | ----------------------------- | -------------------------------------------- |
| Padding                  | Preserve size, handle borders | Keeps edge info, increases cost              |
| Stride                   | Step size of filter           | Larger stride → smaller output               |
| Kernel size              | Region filter covers          | Small = fine details, large = broad patterns |
| Receptive field          | Input region seen by neuron   | Grows with depth                             |
| Depthwise separable conv | Efficiency optimization       | Reduces parameters                           |
| Dilation                 | Expands receptive field       | Captures wider context                       |
| Transposed conv          | Upsampling                    | Used in decoders                             |
| Activation               | Non-linearity                 | Enables complex mappings                     |

---
