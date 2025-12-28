# **1. Long Short-Term Memory (LSTM)**

---

## **Why LSTM Was Needed (Context Recap)**

Vanilla RNNs suffer from:

- **Vanishing gradients** (small weights → information fades)
- **Exploding gradients** (large weights → instability)
- Inability to learn **long-term dependencies**

LSTM was designed specifically to **preserve information over long time horizons** while remaining trainable with gradient descent.

Key design principle:

> Separate _memory_ from _computation_ and control information flow explicitly.

---

## **2. Gated Memory Cell (Core of LSTM)**

### **What Is a Gated Memory Cell?**

An LSTM cell is not just a neuron.
It is a **memory container** with **learnable gates** that decide:

- What to **forget**
- What to **store**
- What to **output**

The memory is called the **cell state** ( c_t ), which flows across time with minimal modification.

This is the critical difference from vanilla RNNs.

---

### **LSTM Cell Components**

An LSTM cell consists of:

1. **Cell state ( c_t )**

   - Long-term memory
   - Additive updates → prevents vanishing gradients

2. **Hidden state ( h_t )**

   - Short-term output
   - Used for predictions and passed to next layer

3. **Three gates** (sigmoid-controlled):

   - Forget gate
   - Input gate
   - Output gate

📌 Gates output values in ([0,1]), acting like **soft switches**.

---

### **LSTM Gate Equations**

Let:

- ( x_t ): input at time (t)
- ( h\_{t-1} ): previous hidden state
- ( c\_{t-1} ): previous cell state

---

### **Forget Gate**

Decides what fraction of previous memory to keep.

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

- ( f_t = 1 ): keep everything
- ( f_t = 0 ): forget everything

This directly addresses **irrelevant long-term memory**.

---

### **Input Gate**

Controls how much new information enters memory.

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

Candidate memory:

$$
\tilde{c}*t = \tanh(W_c [h*{t-1}, x_t] + b_c)
$$

---

### **Cell State Update**

The most important equation in LSTM:

$$
c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t
$$

📌 **Key Insight**:

- Additive update → gradients flow cleanly
- No repeated multiplication → vanishing gradient solved

---

### **Output Gate**

Controls what part of memory is exposed.

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

Hidden state:

$$
h_t = o_t \cdot \tanh(c_t)
$$

---

### **Why LSTM Prevents Vanishing Gradient**

In vanilla RNN:

$$
h_t = \tanh(W_h h_{t-1})
$$

Repeated multiplication:

$$
(W_h)^t \rightarrow 0 \quad \text{or} \quad \infty
$$

In LSTM:

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t
$$

If ( f_t \approx 1 ), gradient flows unchanged.

---

### **LSTM Cell Visualization**

![img](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

---

## **3. LSTM Variants and Advanced Architectures**

---

### **3.1 Bidirectional LSTM (BiLSTM)**

#### **Motivation**

Standard LSTM processes sequences left-to-right, using only **past context**. However, many tasks benefit from **both past and future context**.

Examples:

- **Named Entity Recognition (NER)**: "Bank (institution) decided to open a new branch (location)"
- **Part-of-Speech tagging**: Word's tag depends on surrounding words
- **Machine translation**: Aligning words requires bidirectional understanding

#### **How BiLSTM Works**

BiLSTM runs **two independent LSTM layers**:

1. **Forward LSTM** ($\overrightarrow{LSTM}$): processes left-to-right

   $$
   \overrightarrow{h}_t = \overrightarrow{LSTM}(x_t, \overrightarrow{h}_{t-1})
   $$

2. **Backward LSTM** ($\overleftarrow{LSTM}$): processes right-to-left
   $$
   \overleftarrow{h}_t = \overleftarrow{LSTM}(x_t, \overleftarrow{h}_{t+1})
   $$

Final representation **concatenates** both:

$$
h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]
$$

Output dimension: **2 × hidden_size**

#### **Computational Cost**

- Parameters: **2× that of unidirectional LSTM**
- Computation: **2× forward passes required**
- Memory: **2× hidden states stored**

#### **Training Constraint**

BiLSTM requires the **entire sequence at once** (no streaming):

- ✅ Suitable for batch processing, transcription
- ❌ Not suitable for real-time applications (speech recognition during speaking)

#### **BiLSTM Architecture Diagram**

```
Forward LSTM:  x₁ → x₂ → x₃ → x₄
               ↓    ↓    ↓    ↓
               h₁→  h₂→  h₃→  h₄→

Backward LSTM: x₁ ← x₂ ← x₃ ← x₄
               ↓    ↓    ↓    ↓
               h₁←  h₂←  h₃←  h₄←

Output: [h₁→ h₁←] [h₂→ h₂←] [h₃→ h₃←] [h₄→ h₄←]
```

---

### **3.2 Stacked/Multi-layer LSTMs**

#### **Motivation**

Single LSTM learns **one level of abstraction**. Stacking LSTMs creates a **hierarchy**:

- Layer 1: Low-level patterns (phonemes, character combinations)
- Layer 2: Medium-level patterns (words, sub-phrases)
- Layer 3: High-level semantics (meaning, context)

#### **Architecture**

For $L$ layers, output of layer $l$ becomes input to layer $l+1$:

$$
h_t^{(l)} = \text{LSTM}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)})
$$

Where:

- $h_t^{(0)} = x_t$ (input)
- $h_t^{(L)}$ is final prediction

#### **Depth vs Width Tradeoff**

| Aspect                   | Shallow (1-2 layers) | Deep (4+ layers)            |
| ------------------------ | -------------------- | --------------------------- |
| **Capacity**             | Low                  | High                        |
| **Computational cost**   | Low                  | High                        |
| **Gradient flow**        | Better               | Risk of vanishing gradients |
| **Overfitting risk**     | Low                  | High (needs regularization) |
| **Training time**        | Fast                 | Slow                        |
| **Representation power** | Limited              | Rich hierarchical features  |

#### **Practical Guidelines**

- **2-3 layers**: Standard for most tasks (translation, tagging)
- **4-6 layers**: Deep models (large datasets, complex tasks)
- **Beyond 6 layers**: Rarely beneficial without **residual connections** or **layer normalization**

#### **Recommended Architecture**

```
Input Sequence (batch_size × seq_len × embedding_dim)
    ↓
LSTM Layer 1 (hidden_size=512, bidirectional=True)
    ↓ [output: batch × seq_len × 1024]
Dropout (p=0.5)
    ↓
LSTM Layer 2 (hidden_size=512, bidirectional=True)
    ↓ [output: batch × seq_len × 1024]
Dropout (p=0.5)
    ↓
LSTM Layer 3 (hidden_size=256, bidirectional=True)
    ↓ [output: batch × seq_len × 512]
Output layer
```

---

### **3.3 Gated Recurrent Unit (GRU)**

#### **Motivation**

LSTM has **3 gates** (forget, input, output) and **2 states** (cell, hidden). This is complex.

**GRU simplifies** by:

- **Merging cell and hidden state** into single state $h_t$
- **Reducing to 2 gates** (reset, update)
- **Maintaining gradient flow** benefits
- **30-50% fewer parameters** than LSTM

#### **GRU Equations**

**Reset gate** (what to forget):

$$
r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)
$$

**Update gate** (how much to update):

$$
z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)
$$

**Candidate activation**:

$$
\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)
$$

**Hidden state update**:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

#### **Comparison: LSTM vs GRU**

| Feature                   | LSTM                      | GRU                   |
| ------------------------- | ------------------------- | --------------------- |
| **Gates**                 | 3 (forget, input, output) | 2 (reset, update)     |
| **States**                | 2 (cell, hidden)          | 1 (hidden)            |
| **Parameters**            | $4d(d + m)$               | $3d(d + m)$           |
| **Computation**           | Slower                    | Faster                |
| **Memory**                | Higher                    | Lower                 |
| **Long-term memory**      | Excellent                 | Good                  |
| **Empirical performance** | Slightly better           | Similar on most tasks |

Where $d$ = hidden size, $m$ = input size

#### **When to Use GRU vs LSTM**

✅ **Use GRU when**:

- Limited computational resources
- Hardware constraints (mobile, IoT)
- Speed is critical
- Dataset is small-to-medium

✅ **Use LSTM when**:

- Maximum model capacity needed
- Very long sequences
- Computational resources available
- Need interpretability of cell state

#### **Empirical Finding**

Despite theoretical differences, GRU and LSTM perform **comparably** on most benchmarks. Choice often depends on **engineering constraints** rather than accuracy.

---

### **3.4 Peephole Connections**

#### **Motivation**

Standard LSTM gates depend only on **hidden state and input**:

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

They **ignore the cell state** they're supposed to regulate. This is suboptimal because:

- Forget gate doesn't "see" what it's forgetting
- Input gate doesn't see how full the cell is
- Output gate can't check cell magnitude

**Peephole connections** let gates directly observe the cell state.

#### **Peephole LSTM Equations**

**Forget gate** (with peephole):

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + U_f \odot c_{t-1} + b_f)
$$

**Input gate** (with peephole):

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + U_i \odot c_{t-1} + b_i)
$$

**Output gate** (with peephole):

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + U_o \odot c_t + b_o)
$$

Where:

- $\odot$ is element-wise multiplication
- $U_f, U_i, U_o$ are peephole weight vectors (learned)
- Note: Output gate uses $c_t$ (current), not $c_{t-1}$

#### **Practical Impact**

Peephole connections provide:

- **Modest improvement** on fine-grained timing tasks
- **Better learning dynamics** on some sequential problems
- **Minimal computational overhead** (only element-wise operations)

#### **When Peephole Helps**

✅ **Effective for**:

- Music generation (precise timing matters)
- Digit recognition in images
- Speech recognition
- Tasks requiring **precise temporal alignment**

❌ **Less critical for**:

- Language modeling
- Machine translation
- Most NLP tasks

---

### **3.5 Gradient Flow in LSTM**

#### **Why Gradients Matter**

In vanilla RNN:

$$
\frac{\partial h_t}{\partial h_0} = \prod_{k=1}^{t} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=1}^{t} W_{hh}^T
$$

Repeated matrix multiplication causes:

- **Vanishing gradients**: $(W_{hh})^T$ has spectral radius $< 1$
- **Exploding gradients**: $(W_{hh})^T$ has spectral radius $> 1$

#### **LSTM Solution**

In LSTM, the cell state has **additive updates**:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

Gradient w.r.t. cell state:

$$
\frac{\partial c_t}{\partial c_{t-1}} = \frac{\partial}{\partial c_{t-1}}(f_t \odot c_{t-1} + i_t \odot \tilde{c}_t) = f_t
$$

Key insight:

$$
\frac{\partial L}{\partial c_0} = \frac{\partial L}{\partial c_t} \cdot \prod_{k=1}^{t} f_k
$$

If $f_k \approx 1$ (forget gate open):

- Gradients flow **unchanged** through time
- **No exponential decay or explosion**
- Information preserved over hundreds of steps

#### **Comparison: Gradient Flow Visualization**

```
Vanilla RNN:
┌─────────────┐
│   h₀        │
└────┬────────┘
     │ × W
     ↓
┌─────────────┐
│   h₁        │  ← Gradient is multiplied by W at each step
└────┬────────┘
     │ × W
     ↓
┌─────────────┐
│   h₂        │  ← W^t → 0 or ∞
└──────────────┘

LSTM (cell state path):
┌─────────────┐
│   c₀        │
└────┬────────┘
     │ × f₁ (forget gate)
     ↓
┌─────────────┐
│   c₁        │  ← If f ≈ 1, gradient passes through unchanged
└────┬────────┘
     │ × f₂ (forget gate)
     ↓
┌─────────────┐
│   c₂        │  ← Gradient accumulates additively, not exponentially
└──────────────┘
```

#### **Gradient Flow Equation**

For LSTM, gradient from loss to early cell state:

$$
\frac{\partial L}{\partial c_0} = \frac{\partial L}{\partial c_T} + \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial c_t} \prod_{k=t+1}^{T} f_k
$$

The sum replaces product, preventing vanishing gradients.

#### **Empirical Validation**

✅ LSTM can **learn dependencies 200+ timesteps apart**
✅ Gradient norm remains **stable throughout training**
✅ Forget gate naturally learns to set $f_t \approx 1$ for relevant information

#### **Caveat**

Although LSTM prevents vanishing gradients on the cell state path:

- **Hidden state gradients** can still vanish (hidden state is not purely additive)
- **Deep stacked LSTMs** benefit from layer normalization or residual connections

---

## **3. The Encoder–Decoder Architecture (Motivation)**

---

### **Why Encoder–Decoder Is Needed**

Many problems involve:

- **Variable-length input**
- **Variable-length output**
- No one-to-one alignment

Examples:

- Machine translation
- Speech-to-text
- Text summarization
- Question answering
- Video captioning

Traditional feedforward or RNN models cannot handle this flexibly.

---

### **Core Idea**

> Encode the input sequence into a fixed-size representation, then decode it into another sequence.

---

## **4. Encoder**

---

### **Role of Encoder**

The **encoder** reads the entire input sequence and compresses it into:

- A **context vector**
- Often the final hidden state ( h_T ) (and cell state ( c_T ) in LSTM)

---

### **Encoder Operation**

Given input sequence:
[
(x_1, x_2, ..., x_T)
]

The encoder LSTM computes:
[
(h_1, c_1), (h_2, c_2), ..., (h_T, c_T)
]

Final states:

- ( h_T ): summary of sequence
- ( c_T ): long-term memory

These are passed to the decoder.

---

### **Encoder Diagram**

![Encoder Diagram](https://miro.medium.com/v2/resize:fit:720/format:webp/1*k9xAAAMl-NMTI42MvG--8w.png)

---

### **Subtle Clarification**

❌ Encoder does **not** output predictions
✅ Encoder outputs a **latent representation**

---

## **5. Decoder**

---

### **Role of Decoder**

The decoder:

- Generates output **one step at a time**
- Uses encoder’s final state as initialization
- Predicts next token conditioned on previous outputs

---

### **Decoder Initialization**

Initial states:

$$
h_0^{dec} = h_T^{enc}
$$

$$
c_0^{dec} = c_T^{enc}
$$

---

### **Decoder Recurrence**

At time (t):

$$
(h_t^{dec}, c_t^{dec}) = \text{LSTM}(y_{t-1}, h_{t-1}^{dec}, c_{t-1}^{dec})
$$

Output probability:

$$
p(y_t | y_{<t}, x) = \text{Softmax}(W_o h_t^{dec})
$$

---

### **Training vs Inference**

**Training (Teacher Forcing)**:

- Ground truth previous token is used
- Faster convergence

**Inference**:

- Model’s own prediction is fed back
- Error accumulation possible

---

### **Decoder Diagram**

![decoder](https://miro.medium.com/v2/resize:fit:720/format:webp/1*a4scVhAWcG_6HgGy2Bf0zQ.png)

---

## **6. Putting the Encoder and Decoder Together**

---

![Full architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*5VL80cI_HB1U0hMgxL8GEg.png)

a more detailed architecture is
![Detailed Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/0*376uJu_fc_uR8H3X.png)

### **Full Pipeline**

1. Encoder reads input sequence
2. Final encoder states summarize input
3. Decoder initializes from encoder
4. Decoder generates output sequentially

---

### **Example: Machine Translation**

Input:

```
"I am a student"
```

Encoder produces:

```
Context vector
```

Decoder generates:

```
"Je suis un étudiant"
```

One token at a time.

---

### **Major Limitation of Basic Encoder–Decoder**

❌ Fixed-size context vector
❌ Information bottleneck for long sequences

This led to **Attention Mechanisms**, which allow the decoder to look back at all encoder states.

---

### **Encoder–Decoder with Attention (Preview)**

Instead of using only ( h_T ):

$$
c_t = \sum_{i=1}^{T} \alpha_{ti} h_i^{enc}
$$

Where:

- ( \alpha\_{ti} ) are attention weights

This is the foundation of **Transformers**.

---

## **7. Common Misconceptions (Corrected)**

| Misconception                 | Correction                        |
| ----------------------------- | --------------------------------- |
| LSTM remembers everything     | It learns what to remember        |
| Gates are hard switches       | Gates are differentiable          |
| Encoder outputs prediction    | Encoder outputs representation    |
| LSTM fully solves long memory | Attention does better             |
| LSTM is obsolete              | Still used in low-latency systems |

---

## **8. Applications of LSTM Encoder–Decoder**

- Neural machine translation
- Speech recognition
- Time-series forecasting
- Video captioning
- Text summarization
- Anomaly detection

---

## **9. Summary Table**

| Component       | Purpose              |
| --------------- | -------------------- |
| LSTM Cell       | Controlled memory    |
| Gates           | Regulate information |
| Encoder         | Compress sequence    |
| Decoder         | Generate sequence    |
| Teacher forcing | Stable training      |
| Attention       | Remove bottleneck    |

---

## **Final Takeaway**

LSTM introduced **learnable memory control**, and Encoder–Decoder architectures enabled **sequence-to-sequence learning**, forming the conceptual bridge from RNNs to modern **Transformer-based models**.

---
