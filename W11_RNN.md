# **1. Machine Learning for Sequential Data — Recurrent Neural Networks (RNNs)**

---

## **Why Sequential Data Needs Special Models**

### **Tabular Data → ANN**

In **tabular datasets**, each sample is **independent**:

- Example: student records, customer churn, loan approval.
- Features: age, income, category, etc.
- There is **no temporal or sequential dependency** between rows.

An **Artificial Neural Network (ANN)** works well because:

- Input is a fixed-size vector.
- Samples are IID (independent and identically distributed).
- The model extracts nonlinear feature combinations.

Example:

```
Input: [Age, Income, Education, CreditScore] → Loan Approved?
```

Each row is independent. Shuffling rows does not change meaning.

---

### **Image Data → CNN**

Images are **spatial data (2D or 3D)**:

- Pixel values depend on neighboring pixels.
- Local structure matters.

CNNs exploit:

- **Local receptive fields**
- **Weight sharing**
- **Spatial hierarchy**

Example:

```
Image → Conv Filters → Feature Maps → Dense → Class
```

CNNs **preserve spatial locality**, which ANNs destroy by flattening.

---

### **Sequential / Temporal Data → RNN**

Now consider data where **order matters**:

#### Examples:

- **Speech**: waveform, spectrogram (time × frequency)
- **Stock prices**: price today depends on yesterday
- **Text**: word meaning depends on previous words
- **Video**: frames depend on previous frames
- **Sensor data**: IoT, ECG, EEG
- **Time series forecasting**

Here:

- Samples are **dependent**
- Temporal order is critical
- Shuffling destroys meaning

This is where **Recurrent Neural Networks (RNNs)** are introduced.

---

## **What Is a Recurrent Neural Network (RNN)?**

An **RNN** is a neural network with **memory**.

It processes sequences **one element at a time**, while maintaining a **hidden state** that summarizes past information.

Key idea:

> The output at time (t) depends on the current input **and** previous hidden state.

---

### **RNN Cell Structure**

At each time step (t):

$$h_t = f(W_x x_t + W_h h_{t-1} + b)$$

Where:

- (x_t): input at time (t)
- (h_t): hidden state (memory)
- (W_x): input weight
- (W_h): recurrent weight (shared across time)
- (f): activation function (usually tanh)
- (b): bias

Output:
$$y_t = W_y h_t$$

---

### ** Lecture Example**

w3 is the reccurent weight between time steps.

```
y1 = w1x1 + b
y2 = w2x2 + w3y1 + b
y3 = w4x3 + w3y2 + b
```

interpretation:

- (w_3) is the **recurrent weight**
- It is **shared across all time steps**
- This weight causes gradient issues later

More formally:

$$h_1 = W_x x_1 + b$$

$$h_2 = W_x x_2 + W_h h_1 + b$$

$$h_3 = W_x x_3 + W_h h_2 + b$$

This **weight sharing across time** is the defining feature of RNNs.

---

## **Unrolling the RNN**

To understand training, RNNs are **unrolled** over time:

```
x1 → [RNN] → h1 → y1
x2 → [RNN] → h2 → y2
x3 → [RNN] → h3 → y3
```

All RNN cells are **copies** sharing the same parameters.

📷 Example visualization:

[RNN Architecture diagram](https://www.researchgate.net/profile/Rishikesh-Gawde/publication/351840108/figure/fig1/AS:1027303769374723@1621939713840/Fig-3-RNN-A-recurrent-neural-network-RNN-is-a-class-of-artificial-neural-networks.ppm)

---

## **Backpropagation Through Time (BPTT)**

Training RNNs uses **Backpropagation Through Time**.

Steps:

1. Forward pass through all time steps.
2. Compute loss at final or each time step.
3. Backpropagate gradients **backward through time**.

Mathematically:
$$\frac{\partial L}{\partial W_h} = \sum_{t} \frac{\partial L}{\partial h_t} \prod_{k=1}^{t} \frac{\partial h_k}{\partial h_{k-1}}$$

This product term causes instability.

---

## **Major Issues in Vanilla RNNs**

### **1. No Parallelism**

- Each step depends on previous step.
- Cannot parallelize across time.
- Slow training compared to CNNs/Transformers.

---

### **2. Vanishing Gradient Problem**

If:
$$|W_h| < 1$$

Then repeated multiplication:
$$(W_h)^t \rightarrow 0$$

Effect:

- Gradients shrink exponentially.
- Early time steps stop learning.
- Model forgets long-term dependencies.

This is why RNNs struggle with long sequences.

---

### **3. Exploding Gradient Problem**

If:
$$|W_h| > 1$$

Then:
$$(W_h)^t \rightarrow \infty$$

Effect:

- Gradients explode.
- Training diverges.
- Loss becomes NaN.

Fixes:

- Gradient clipping
- Proper initialization

---

### **Why Early Information Is Forgotten**

Your lecture point is correct:

- Only last layers receive meaningful gradients.
- Earlier contributions fade due to vanishing gradients.
- This limits RNN memory.

---

## **Applications of Vanilla RNNs**

- Short sequences
- Simple time-series
- Basic sequence classification

But for **long-term dependencies**, vanilla RNNs are insufficient.

---

# **2. Gated RNNs — LSTM and GRU**

---

## **Why Gated RNNs Were Introduced**

To fix:

- Vanishing gradients
- Memory loss
- Long dependency failure

Solution:

> Introduce **gates** to control information flow.

Gated RNNs learn:

- What to remember
- What to forget
- What to output

---

## **Long Short-Term Memory (LSTM)**

### **Core Idea**

LSTM introduces a **cell state** (c_t) that flows unchanged across time.

This allows gradients to pass without vanishing.

---

### **LSTM Architecture**

An LSTM has:

- Input gate
- Forget gate
- Output gate
- Cell state

[LSTM Architecture diagram](https://www.researchgate.net/profile/Ahmed-Elkaseer/publication/356018554/figure/fig1/AS:1088159563677697@1636448865987/A-Long-short-term-memory-LSTM-unit-architecture.png)

---

### **LSTM Equations**

Forget gate:
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

Input gate:
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

Candidate memory:
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

Cell state update:
$$c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t$$

Output gate:
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

Hidden state:
$$h_t = o_t \cdot \tanh(c_t)$$

---

### **Why LSTM Solves Vanishing Gradient**

- Cell state has additive updates
- Gradients flow through (c_t) almost unchanged
- Gates learn to protect useful information

---

## **Gated Recurrent Unit (GRU)**

GRU is a **simplified LSTM**.

Differences:

- No separate cell state
- Fewer gates
- Faster training

---

[GRU Architecture diagram](https://www.researchgate.net/publication/370683092/figure/fig2/AS:11431281207716844@1701313562353/TheArchitecture-of-the-gated-recurrent-unit-GRU-cell.png)

### **GRU Equations**

Update gate:
$$z_t = \sigma(W_z [h_{t-1}, x_t])$$

Reset gate:
$$r_t = \sigma(W_r [h_{t-1}, x_t])$$

Candidate state:
$$\tilde{h}_t = \tanh(W [r_t \cdot h_{t-1}, x_t])$$

Final state:
$$h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t$$

---

### **LSTM vs GRU**

| Feature        | LSTM                              | GRU              |
| -------------- | --------------------------------- | ---------------- |
| Gates          | 3                                 | 2                |
| Cell state     | Yes                               | No               |
| Parameters     | More                              | Fewer            |
| Training speed | Slower                            | Faster           |
| Performance    | Slightly better on long sequences | Often comparable |

---

## **Applications of Gated RNNs**

- Speech recognition
- Machine translation
- Video analysis
- Time-series forecasting
- NLP (pre-transformer era)

---

## **Limitations of RNNs (Even Gated Ones)**

- Sequential computation (slow)
- Difficult to scale
- Long-range dependency still limited
- Led to Transformers

---

## **Final Conceptual Summary**

| Data Type            | Model        |
| -------------------- | ------------ |
| Tabular              | ANN          |
| Images               | CNN          |
| Sequential           | RNN          |
| Long Sequential      | LSTM / GRU   |
| Very Long + Parallel | Transformers |

---

