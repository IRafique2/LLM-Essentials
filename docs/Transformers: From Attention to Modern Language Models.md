# Day 08 — Transformers: From Attention to Modern Language Models

> **Transformers changed how machines understand language by removing recurrence and relying entirely on attention.**


---

## 1. What Is a Transformer?

A **Transformer** is a neural network architecture designed to solve **sequence-to-sequence (Seq2Seq)** problems such as:

* Machine Translation
* Text Summarization
* Question Answering
* Dialogue Generation

Originally introduced in the paper **"Attention Is All You Need" (Vaswani et al., 2017)**, Transformers were created to overcome the limitations of RNN-based models.

### Why the Name *Transformer*?

Transformers **transform an input sequence into an output sequence** by learning relationships between tokens — *without using recurrence or convolution*.

---

## 2. Why Transformers Were Needed

Before Transformers, NLP relied on:

* **RNNs** → slow, sequential, vanishing gradients
* **LSTMs / GRUs** → better memory but still sequential and hard to parallelize

### Core Problems with RNN-based Models

*  Cannot process sequences in parallel
*  Training is slow
*  Struggle with very long dependencies

### Key Insight Behind Transformers

> **Sequence understanding does not require recurrence — it requires attention.**

Transformers achieve **full parallelization** and **global context modeling** using **attention mechanisms**.

---

## 3. Transformer Model Overview

The Transformer follows the classic **Encoder–Decoder architecture**, but with a critical difference:

>  No RNNs
 No CNNs
>  Only Attention + Feed-Forward Networks

### High-Level Structure

```
Input Sentence → Encoder Stack → Contextual Representations
                                      ↓
                              Decoder Stack → Output Sentence
```

* **Encoder** → Understands the input
* **Decoder** → Generates the output

The original Transformer uses:

* **6 Encoder layers**
* **6 Decoder layers**

(But this number is fully configurable.)

---

## 4. Transformer Encoder: Detailed Workflow

The **Encoder** converts input tokens into **context-aware vector representations**.

### Encoder Layer Components

Each encoder layer contains:

1. Multi-Head Self-Attention
2. Feed-Forward Neural Network
3. Residual Connections
4. Layer Normalization

---

### 4.1 Step 1 — Input Embeddings

* Input text is tokenized (words/subwords)
* Each token is mapped to a dense vector

Example:

```
"How are you ?" → [e1, e2, e3, e4]
```

Each embedding typically has **512 dimensions** (base Transformer).

---

### 4.2 Step 2 — Positional Encoding

Transformers lack recurrence, so **order information must be injected manually**.

#### Why Positional Encoding?

The sentences:

* "She did not win the award, she was satisfied"
* "She did win the award, she was not satisfied"

Contain the same words — but **different meanings**.

#### Sinusoidal Positional Encoding

For position `pos` and dimension `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Properties:

* Unique encoding per position
* Generalizes to longer sequences
* Encodes relative distances

---

### 4.3 Step 3 — Multi-Head Self-Attention

Self-attention allows **each word to attend to every other word** in the sequence.

#### Query, Key, Value

For each token embedding `x`:

```
Q = xW_Q
K = xW_K
V = xW_V
```

#### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

* Similarity → dot product
* Scaling → stabilizes gradients
* Softmax → probability distribution

#### Intuition

> *The meaning of a word is the weighted sum of the words it pays attention to.*

---

### Multi-Head Attention

Instead of one attention operation, Transformers use **multiple heads**:

```
head_i = Attention(QW_i, KW_i, VW_i)
MultiHead = Concat(head_1, ..., head_h)W_O
```

Benefits:

* Different heads learn different relationships
* Syntax, semantics, position, alignment

---

### 4.4 Residual Connections & Layer Normalization

Each sub-layer applies:

```
Output = LayerNorm(x + Sublayer(x))
```

Why?

* Prevent vanishing gradients
* Preserve information
* Stabilize deep training

---

### 4.5 Feed-Forward Neural Network

Applied **independently to each position**:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

Acts as:

* Feature transformer
* Non-linear refinement

---

### 4.6 Encoder Output

Final encoder output:

* A sequence of vectors
* Each vector knows **what the word means in context**

This output is passed to the decoder.

---

## 5. Transformer Decoder: Detailed Workflow

The **Decoder** generates output tokens **autoregressively**.

### Decoder Layer Components

Each decoder layer contains:

1. Masked Self-Attention
2. Encoder–Decoder (Cross) Attention
3. Feed-Forward Network
4. Residual + LayerNorm

---

### 5.1 Step 1 — Output Embeddings

Decoder input begins with:

```
<START> token
```

Then continues with previously generated tokens.

---

### 5.2 Step 2 — Positional Encoding

Same sinusoidal positional encoding as encoder.

---

### 5.3 Masked Self-Attention

Prevents attending to **future tokens**.

Why?

> The model must not cheat during generation.

Masking sets attention scores of future positions to `-∞`.

---

### 5.4 Cross Attention (Encoder–Decoder Attention)

Here:

* **Queries** → Decoder
* **Keys & Values** → Encoder output

This allows the decoder to:

> Focus on relevant parts of the input sequence.

---

### 5.5 Feed-Forward Network

Same structure as encoder FFN.

---

### 5.6 Linear Layer & Softmax

Final step:

```
logits = DecoderOutput × W_vocab
probabilities = softmax(logits)
```

* Vocabulary size = number of classes
* Highest probability token is selected

---

## 6. Training vs Inference

### Teacher Forcing (Training)

* Ground-truth tokens fed into decoder
* Faster convergence

### Inference

* Model feeds its own predictions
* Continues until `<END>` token

---

## 7. Transformer Variants

### Encoder-Only

* **BERT**, RoBERTa
* Tasks: classification, QA, MLM

### Decoder-Only

* **GPT**, ChatGPT
* Tasks: text generation

### Encoder–Decoder

* **T5**, BART, Pegasus
* Tasks: translation, summarization

---

## 8. Why Transformers Matter

Transformers enable:

* Parallel training
* Long-range dependency modeling
* Scalable architectures

> **All modern LLMs are Transformer-based.**

---

## 9. Key Takeaways

* Attention replaces recurrence
* Positional encoding injects order
* Multi-head attention captures diverse relations
* Encoder understands, decoder generates

---

## Acknowledgement

This content is compiled from lecture slides, research papers, and educational resources and is intended solely for learning and clarification purposes.
