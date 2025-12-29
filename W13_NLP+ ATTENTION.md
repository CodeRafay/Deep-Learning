# **1. Introduction to Natural Language Processing (NLP)**

---

## **1.1 What Is Natural Language Processing**

Natural Language Processing (NLP) is a subfield of **Artificial Intelligence (AI)** and **Computational Linguistics** that focuses on enabling machines to **understand, interpret, generate, and interact with human language** in a meaningful way.

Human language is:

- Ambiguous
- Context-dependent
- Noisy
- Evolving
- Governed by implicit rules rather than strict formal grammars

The core challenge of NLP is that **natural language was designed for human communication, not machine processing**.

---

## **1.2 Why NLP Is Difficult**

Unlike structured data:

- Language has **multiple meanings for the same word**
- Word order matters
- Meaning changes with context
- Grammar rules are inconsistent
- Cultural and pragmatic knowledge affects interpretation

Example:

> “I saw the man with the telescope.”

This sentence has multiple valid interpretations, which machines must learn to disambiguate.

---

## **1.3 Levels of Language Analysis in NLP**

NLP is traditionally divided into multiple linguistic levels:

### **1.3.1 Lexical Analysis**

Focuses on:

- Words
- Tokens
- Vocabulary
- Morphology

Tasks include:

- Tokenization
- Stemming
- Lemmatization
- Stop-word removal

---

### **1.3.2 Syntactic Analysis**

Focuses on sentence structure and grammar.

Tasks include:

- Parsing
- Part-of-Speech (POS) tagging
- Dependency trees
- Constituency parsing

Example:

> “The cat sat on the mat”

POS tagging:

- The (DET)
- cat (NOUN)
- sat (VERB)

---

### **1.3.3 Semantic Analysis**

Focuses on meaning.

Tasks include:

- Word sense disambiguation
- Named Entity Recognition (NER)
- Semantic role labeling

---

### **1.3.4 Pragmatic Analysis**

Focuses on meaning beyond literal words.

Example:

> “Can you open the window?”

Literally a question, pragmatically a request.

---

## **1.4 Core NLP Tasks**

| Category          | Examples                           |
| ----------------- | ---------------------------------- |
| Classification    | Sentiment analysis, spam detection |
| Sequence labeling | POS tagging, NER                   |
| Generation        | Text generation, summarization     |
| Translation       | Machine translation                |
| Retrieval         | Search engines, question answering |

---

## **1.5 NLP Pipeline (Classical)**

![pipeline](https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/_images/nlp-pipeline.png)

Typical pipeline:

1. Text preprocessing
2. Feature extraction
3. Model training
4. Inference and evaluation

---

## **1.6 Text Representation Techniques**

Machines cannot directly understand text. It must be converted into numbers.

### **1.6.1 Bag of Words (BoW)**

- Counts word occurrences
- Ignores word order
- High dimensional and sparse

---

### **1.6.2 TF-IDF**

Weights words by importance:

$$
\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \log \left(\frac{N}{DF(w)}\right)
$$

---

### **1.6.3 Word Embeddings**

Dense vector representations capturing semantic meaning.

Examples:

- Word2Vec
- GloVe
- FastText

Property:

> Similar words have similar vectors.

---

## **1.7 Statistical vs Neural NLP**

### **Traditional NLP**

- Rule-based systems
- Probabilistic models (HMMs, CRFs)
- Heavy feature engineering

### **Neural NLP**

- Uses neural networks
- Learns features automatically
- Dominates modern NLP

---

## **1.8 Modern NLP Landscape**

Key architectures:

- RNNs
- LSTMs
- GRUs
- Transformers

Modern NLP systems rely heavily on **attention mechanisms**, which solved many limitations of earlier models.

---

## **1.9 Common Misconceptions (Corrected)**

| Misconception                           | Correction                                |
| --------------------------------------- | ----------------------------------------- |
| NLP is only about text                  | It includes speech and multimodal data    |
| Grammar rules are enough                | Statistical learning is essential         |
| Word embeddings store meaning perfectly | They capture usage patterns               |
| NLP understands language like humans    | It models correlations, not understanding |

---

## **1.10 Applications of NLP**

- Search engines
- Chatbots
- Virtual assistants
- Document classification
- Machine translation
- Speech recognition
- Legal and medical text analysis

---

## **1.11 Key Takeaway**

NLP is fundamentally about **bridging the gap between unstructured human language and structured machine computation**, and its evolution is tightly coupled with advances in machine learning and deep learning.

---

# **2. Attention Mechanism**

---

## **2.1 Motivation for Attention**

Early sequence-to-sequence models used a **fixed-length context vector** to represent the entire input.

Problem:

- Long sequences cause information loss
- Bottleneck limits performance
- Decoder cannot selectively focus on relevant input parts

Attention was introduced to allow the model to **dynamically focus on relevant information**.

---

## **2.2 Core Idea of Attention**

> Instead of compressing the input into a single vector, allow the model to look at all input representations and assign importance scores.

---

## **2.3 Attention in Encoder–Decoder Models**

At decoding step (t), compute a context vector:

$$
c_t = \sum_{i=1}^{T} \alpha_{ti} h_i
$$

Where:

- $h_i$: encoder hidden states
- $\alpha_{ti}$: attention weights

---

## **2.4 Computing Attention Weights**

### **Alignment Score**

Several scoring functions exist:

**Dot Product**

$$
e_{ti} = h_t^{dec} \cdot h_i^{enc}
$$

**General**

$$
e_{ti} = h_t^{dec^T} W h_i^{enc}
$$

**Additive (Bahdanau Attention)**

$$
e_{ti} = v^T \tanh(W_1 h_i + W_2 h_t)
$$

---

### **Softmax Normalization**

$$
\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{T} \exp(e_{tj})}
$$

Properties:

- Sum to 1
- Differentiable
- Interpretable

---

## **2.5 Attention Visualization**

![attention](https://media.geeksforgeeks.org/wp-content/uploads/20251029093447681289/string_constant_pool_5.webp)

---

## **2.6 Why Attention Works**

- Removes information bottleneck
- Enables long-range dependencies
- Improves alignment
- Improves interpretability

---

## **2.7 Types of Attention**

- Global attention
- Local attention
- Soft attention
- Hard attention (non-differentiable)
- Additive vs multiplicative attention

---

## **2.8 Misconceptions About Attention**

| Misconception                      | Reality                         |
| ---------------------------------- | ------------------------------- |
| Attention equals explanation       | It is a learned weighting       |
| Attention replaces RNNs            | Initially, it complemented them |
| Attention always improves accuracy | Depends on task                 |

---

## **2.9 Applications of Attention**

- Machine translation
- Image captioning
- Speech recognition
- Question answering
- Text summarization

---

# **3. Self-Attention Mechanism**

---

## **3.1 What Is Self-Attention**

Self-attention is a mechanism where **a sequence attends to itself**, enabling each token to interact with every other token in the same sequence.

This removes recurrence entirely.

---

## **3.2 Why Self-Attention Was Needed**

Limitations of RNNs:

- Sequential computation
- Poor parallelization
- Long dependency paths

Self-attention allows:

- Full parallelism
- Direct token-to-token interaction
- Global context modeling

---

## **3.3 Query, Key, Value (QKV) Framework**

Each input token (x_i) is projected into:

- Query (Q)
- Key (K)
- Value (V)

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

---

## **3.4 Scaled Dot-Product Attention**

$$
\text{Attention}(Q, K, V) =
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Scaling by $\sqrt{d_k}$ prevents softmax saturation.

---

## **3.5 Self-Attention Diagram**

![self-attention](https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/summary.png)

---

## **3.6 Why Self-Attention Is Powerful**

- Captures syntactic and semantic relationships
- Models long-range dependencies efficiently
- Language-agnostic
- Highly parallelizable

---

## **3.7 Issues in Self-Attention**

While self-attention is powerful, it has several limitations:

### **Issue 1: No Notion of Position/Order**

Self-attention treats input as an unordered set. The sentence "The cat ate the mouse" produces the same attention as "The mouse ate the cat" without positional information.

### **Issue 2: Single Representation Space**

Using a single attention mechanism limits the model's ability to capture different types of relationships simultaneously (e.g., syntactic vs semantic).

### **Issue 3: Lack of Non-Linearity**

Self-attention is essentially a weighted average operation (linear). Without non-linear transformations, it cannot learn complex polynomial relationships or intricate behavioral patterns.

### **Issue 4: Information Leakage in Autoregressive Tasks**

During generation tasks (like language modeling), the model should only attend to previous tokens, not future ones. Without masking, the model "cheats" by seeing future information.

**ASCII Visualization of Attention Flow:**

```
Input:    [The]  [cat]  [sat]  [on]  [mat]
           |      |      |      |     |
           v      v      v      v     v
         +---------------------------+
         |   Self-Attention Layer    |
         |  (attends to all tokens)  |
         +---------------------------+
           |      |      |      |     |
           v      v      v      v     v
Output:  [The'] [cat'] [sat'] [on'] [mat']
```

---

## **3.8 Multi-Head Self-Attention**

### **Solution to Single Representation Issue**

Instead of one attention operation, multiple heads are used in parallel. This allows the model to attend to information from different representation subspaces at different positions.

### **Architecture**

With **h** attention heads (typically h = 4, 8, or 16):

1. Each head has its own learned projection matrices: $W_Q^i, W_K^i, W_V^i$
2. Each head computes attention independently
3. Outputs are concatenated and projected

$$
head_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)
$$

Each head learns different relationships.

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(head_1,\dots,head_h)W^O
$$

Where $W^O$ is the output projection matrix.

### **Example: 4 Attention Heads**

```
Input (d_model = 512)
         |
    Split into 4 heads (d_k = 128 each)
    /    |    |    \
  Head1 Head2 Head3 Head4
    |    |    |    |
  (128) (128) (128) (128)
    \    |    |    /
      Concatenate
         |
    Linear (W^O)
         |
   Output (512)
```

**ASCII Diagram of Multi-Head Attention:**

```
      Q           K           V
      |           |           |
   +--+--+     +--+--+     +--+--+
   |  |  |     |  |  |     |  |  |
  Q1 Q2 Q3    K1 K2 K3    V1 V2 V3  (Split)
   |  |  |     |  |  |     |  |  |
   +--+--+     +--+--+     +--+--+
   |  |  |     |  |  |     |  |  |
Attn1 Attn2 Attn3 (3 heads)
   |  |  |
   +--+--+
      |
  Concat + Linear
      |
   Output
```

### **Benefits**

- **Diversity**: Each head can focus on different aspects (syntactic structure, semantic meaning, positional relationships)
- **Expressiveness**: Captures multiple types of dependencies simultaneously
- **Robustness**: Redundancy across heads improves learning

**Example Use Cases:**

- Head 1: Subject-verb relationships
- Head 2: Long-range dependencies
- Head 3: Local context
- Head 4: Semantic similarity

---

## **3.9 Positional Encoding**

### **Solution to Position-Agnostic Issue**

Self-attention has no inherent notion of order. Without positional information, permuting the input produces identical outputs.

**Problem:**

- "The cat chased the dog" = "The dog chased the cat" (without position)

**Solution:**
Add positional information to input embeddings.

---

### **Types of Positional Encoding**

#### **3.9.1 Absolute Positional Encoding**

Each position gets a unique encoding added to its embedding.

**Sinusoidal Positional Encoding (Transformer Original):**

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

Where:

- $pos$: position in sequence
- $i$: dimension index
- $d$: embedding dimension

**Properties:**

- Deterministic (not learned)
- Allows extrapolation to longer sequences
- Each dimension has a different wavelength
- Relative positions can be represented as linear functions

**Visualization:**

```
Position:  0    1    2    3    4
Dim 0:    0.0  0.84 0.91 0.14 -0.76  (sin)
Dim 1:    1.0  0.54 -0.42 -0.99 -0.65 (cos)
Dim 2:    0.0  0.10 0.20 0.30 0.39   (sin, slower)
Dim 3:    1.0  0.99 0.98 0.95 0.92   (cos, slower)
...
```

**ASCII Diagram:**

```
Word Embedding:     [0.2, 0.5, -0.1, 0.8, ...]
                              +
Positional Encoding: [0.0, 1.0, 0.01, 0.99, ...]
                              |
                              v
Final Input:        [0.2, 1.5, -0.09, 1.79, ...]
```

#### **3.9.2 Relative Positional Encoding**

Encodes relative distances between tokens rather than absolute positions.

**Advantages:**

- Better generalization to unseen sequence lengths
- Focus on relative relationships ("word A is 3 positions before word B")
- Used in models like Transformer-XL, T5

**Methods:**

1. **Learned Relative Embeddings**: Add learnable bias based on relative distance

$$
e_{ij} = \frac{(x_i W_Q)(x_j W_K)^T}{\sqrt{d_k}} + r_{i-j}
$$

Where $r_{i-j}$ is a learned relative position bias.

2. **Clipped Relative Positions**: Clip maximum distance

```
Relative Distance Matrix:
     0  1  2  3  4
0 [  0  1  2  3  4]
1 [ -1  0  1  2  3]
2 [ -2 -1  0  1  2]
3 [ -3 -2 -1  0  1]
4 [ -4 -3 -2 -1  0]
```

---

### **How Positional Encoding Enables Key Benefits**

#### **1. Resolves Generalization Issue**

- Sinusoidal encoding allows handling sequences longer than training
- Pattern continues infinitely with deterministic formula

#### **2. Resolves Temporal Dependencies**

- Adds explicit order information
- Model can distinguish "before" and "after"
- Maintains causality in sequence

#### **3. Enables Parallelism**

Without positional encoding, we'd need sequential processing (like RNNs) to maintain order.

**With Positional Encoding:**

```
All positions computed simultaneously
    ↓         ↓         ↓         ↓
[Token1] [Token2] [Token3] [Token4]
    ↓         ↓         ↓         ↓
  [PE1]     [PE2]     [PE3]     [PE4]
    ↓         ↓         ↓         ↓
    +         +         +         +
    ↓         ↓         ↓         ↓
[Input1] [Input2] [Input3] [Input4]
         All processed in parallel!
```

**Without (like RNN):**

```
[Token1] → [Token2] → [Token3] → [Token4]
(sequential, slow)
```

#### **4. Reduces Computation**

- No recurrence = O(1) sequential operations (vs O(n) for RNNs)
- Matrix operations are GPU-friendly
- Training time reduced significantly

**Complexity Comparison:**

| Model     | Sequential Ops | Parallel Ops | Max Path Length |
| --------- | -------------- | ------------ | --------------- |
| RNN       | O(n)           | O(n)         | O(n)            |
| Self-Attn | O(1)           | O(n²)        | O(1)            |

---

## **3.10 Adding Non-Linearities**

### **Solution to Linearity Problem**

Self-attention performs weighted averaging (linear operation):

$$
\text{Output} = \text{softmax}(QK^T)V
$$

This is fundamentally linear and cannot model complex, non-linear relationships.

### **Problem Example**

```
Linear: y = ax + b (straight line)
Can't learn: y = x² or y = sigmoid(x)
```

### **Solution: Add Non-Linear Activations**

#### **Feed-Forward Networks (FFN)**

After each attention layer, apply:

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

Or with other activations:

$$
\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
$$

Where $\sigma$ can be:

- **ReLU**: $\max(0, x)$
- **GELU**: $x \cdot \Phi(x)$ (Gaussian Error Linear Unit)
- **Sigmoid**: $\frac{1}{1 + e^{-x}}$
- **Swish**: $x \cdot \sigma(x)$

### **Complete Transformer Block**

```
      Input
        |
  Multi-Head Attention
        |
    Add & Norm (residual)
        |
   Feed-Forward Network  ← Non-linearity here!
   (Linear → ReLU → Linear)
        |
    Add & Norm (residual)
        |
      Output
```

### **Why Non-Linearity Matters**

1. **Complex Patterns**: Can model polynomial relationships
2. **Feature Interactions**: Non-linear combinations of features
3. **Universal Approximation**: With enough layers, can approximate any function
4. **Hierarchical Learning**: Each layer learns progressively complex patterns

**Example:**

```
Without non-linearity:
  Stack 10 linear layers = 1 linear layer (no benefit)

With non-linearity:
  Stack 10 layers = very complex function
  (can learn intricate patterns)
```

---

## **3.11 Masked Decoding**

### **Solution to Information Leakage Problem**

**Problem During Training/Generation:**

In autoregressive tasks (language modeling, translation), the model should predict token $t$ using only tokens $1, 2, \dots, t-1$.

Without masking:

```
Predicting position 2, but can see ALL positions:
  ↓
[The] [cat] [sat] [on] [mat]
  ↑     ↑     ↑     ↑     ↑
  └─────┴─────┴─────┴─────┘
      Attention sees everything!
      (cheating by seeing future)
```

### **Causal/Autoregressive Masking**

**Attention Matrix Without Masking:**

```
Attention scores (all visible):
       The  cat  sat  on  mat
The  [0.2  0.3  0.1  0.2  0.2]
cat  [0.1  0.4  0.2  0.2  0.1]
sat  [0.1  0.2  0.3  0.3  0.1]
on   [0.2  0.1  0.2  0.4  0.1]
mat  [0.1  0.2  0.2  0.2  0.3]
```

**With Causal Masking (Upper Triangle → -∞ before softmax):**

```
Masked Attention:
       The  cat  sat  on  mat
The  [0.2  -∞   -∞   -∞   -∞ ]  ← only sees "The"
cat  [0.3  0.7  -∞   -∞   -∞ ]  ← sees "The", "cat"
sat  [0.2  0.3  0.5  -∞   -∞ ]  ← sees up to "sat"
on   [0.1  0.2  0.3  0.4  -∞ ]  ← sees up to "on"
mat  [0.1  0.2  0.2  0.2  0.3]  ← sees all previous
```

**After Softmax (masked values become 0):**

```
       The  cat  sat  on  mat
The  [1.0  0.0  0.0  0.0  0.0]
cat  [0.3  0.7  0.0  0.0  0.0]
sat  [0.2  0.3  0.5  0.0  0.0]
on   [0.1  0.2  0.3  0.4  0.0]
mat  [0.1  0.2  0.2  0.2  0.3]
```

### **Implementation**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
$$

Where mask $M$ is:

$$
M_{ij} = \begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}
$$

### **ASCII Visualization of Masked Self-Attention**

```
Decoder at position 3 (predicting "on"):

  Input: [The] [cat] [sat] [?]
                             ↑
                    Only attends to:
         [The] [cat] [sat]
           ↓     ↓     ↓
         +-------------+
         | Attention   |  ← Masked
         +-------------+
               ↓
         Predict: [on]
```

### **Attention Matrix with Upper Triangular Masking**

**Visual: Which positions can attend to which**

```
Query\Key   The   cat   sat   on    mat
-----------------------------------------
The      [  ✓    ✗    ✗    ✗    ✗ ]
cat      [  ✓    ✓    ✗    ✗    ✗ ]
sat      [  ✓    ✓    ✓    ✗    ✗ ]
on       [  ✓    ✓    ✓    ✓    ✗ ]
mat      [  ✓    ✓    ✓    ✓    ✓ ]

Legend:
  ✓ = Can attend (visible/allowed)
  ✗ = Masked (hidden/set to -∞)
```

**Upper Triangular Masking Pattern:**

```
Lower Triangular (allowed) = seen
Upper Triangular (masked) = hidden

      0   1   2   3   4
    ┌────┬────┬────┬────┬──────┐
  0 │ ✓  │ ✗  │ ✗  │ ✗  │ ✗  │
    ├────┼────┼────┼────┼──────┤
  1 │ ✓  │ ✓  │ ✗  │ ✗  │ ✗  │
    ├────┼────┼────┼────┼──────┤
  2 │ ✓  │ ✓  │ ✓  │ ✗  │ ✗  │
    ├────┼────┼────┼────┼──────┤
  3 │ ✓  │ ✓  │ ✓  │ ✓  │ ✗  │
    ├────┼────┼────┼────┼──────┤
  4 │ ✓  │ ✓  │ ✓  │ ✓  │ ✓  │
    └────┴────┴────┴────┴──────┘

Position can only attend to:
  Pos 0: itself only
  Pos 1: itself + pos 0
  Pos 2: itself + pos 0,1
  Pos 3: itself + pos 0,1,2
  Pos 4: all previous positions
```

**Numerical Example: Attention Scores Before & After Masking**

```
Before Masking (all visible):
       The    cat    sat    on     mat
The  [0.8   0.5   0.3   0.2   0.1]
cat  [0.2   0.7   0.4   0.3   0.1]
sat  [0.1   0.2   0.6   0.4   0.2]
on   [0.2   0.1   0.3   0.5   0.3]
mat  [0.1   0.2   0.2   0.2   0.4]

Apply Mask (Upper Triangular → -∞):
       The    cat    sat    on     mat
The  [0.8   -∞    -∞    -∞    -∞  ]
cat  [0.2   0.7   -∞    -∞    -∞  ]
sat  [0.1   0.2   0.6   -∞    -∞  ]
on   [0.2   0.1   0.3   0.5   -∞  ]
mat  [0.1   0.2   0.2   0.2   0.4]

After Softmax (masked positions become 0):
       The    cat    sat    on     mat
The  [1.0   0.0   0.0   0.0   0.0]  (only attends to itself)
cat  [0.22  0.78  0.0   0.0   0.0]  (attends to The, cat)
sat  [0.10  0.15  0.75  0.0   0.0]  (attends to The, cat, sat)
on   [0.17  0.10  0.25  0.48  0.0]  (attends to The, cat, sat, on)
mat  [0.10  0.20  0.20  0.20  0.30] (attends to all)
```

### **When to Use Masking**

| Component                     | Masking? | Reason                            |
| ----------------------------- | -------- | --------------------------------- |
| **Encoder**                   | No       | Can see full input context        |
| **Decoder (self-attention)**  | Yes      | Autoregressive generation         |
| **Decoder (cross-attention)** | No       | Can attend to full encoder output |

### **Why Masking Works**

1. **Prevents Cheating**: Forces model to learn true sequential dependencies
2. **Matches Inference**: Training and testing conditions are identical
3. **Causal Structure**: Maintains temporal causality
4. **No Information Leakage**: Each prediction is independent of future tokens

**Training vs Testing:**

```
Training (with masking):
  Predict all positions in parallel
  But each masked to only see past

Testing/Inference:
  Generate one token at a time
  Naturally only sees past tokens

Both scenarios identical due to masking!
```

---

## **3.12 Common Misconceptions**

| Misconception                            | Correction                                  |
| ---------------------------------------- | ------------------------------------------- |
| Self-attention ignores order             | Positional encoding adds order              |
| Self-attention is attention              | It is a special case                        |
| Transformers have memory                 | Memory is implicit                          |
| Masking is only for training             | Used in both training and inference         |
| All attention heads learn the same thing | Each head specializes in different patterns |

---

## **3.13 Applications of Self-Attention**

- Transformers
- BERT
- GPT
- Vision Transformers
- Multimodal models

---

## **Final Unified Takeaway**

NLP evolved from rule-based systems to neural architectures, and the introduction of **attention and self-attention fundamentally changed how machines model language**, enabling scalable, interpretable, and highly effective language understanding systems.

---
