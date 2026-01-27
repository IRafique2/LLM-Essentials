# Day 13 — Transformer-Based Models (BERT & GPT)



## 1. Why Transformer-Based Models?

Before Transformers, sequence models such as **RNNs** and **LSTMs** were widely used for NLP tasks. However, they suffered from:

* Sequential processing (slow training)
* Difficulty capturing long-range dependencies
* Limited parallelization

Transformers solved these issues using **self-attention**, enabling:

* Parallel computation
* Better global context modeling
* Scalability to very large datasets

---

## 2. What Is a Transformer Base Model?

A **Transformer base model** is a large neural network built using Transformer blocks and **pre-trained on massive text corpora** to learn language structure, grammar, and semantics.

Key characteristics:

* Uses **self-attention** instead of recurrence
* Pre-trained in an unsupervised or self-supervised manner
* Fine-tuned or prompted for downstream tasks

Two dominant families:

* **Encoder-only** → BERT
* **Decoder-only** → GPT

---

## 3. LSTM vs Transformer (Conceptual Comparison)

| Aspect         | LSTM       | Transformer                  |
| -------------- | ---------- | ---------------------------- |
| Processing     | Sequential | Parallel                     |
| Context        | Limited    | Global                       |
| Directionality | Partial    | Fully bidirectional / causal |
| Speed          | Slow       | Fast                         |

Transformers enable **simultaneous token processing**, making them ideal for large-scale language modeling.

---

## 4. BERT — Bidirectional Encoder Representations from Transformers

### 4.1 What Is BERT?

**BERT** is an **encoder-only Transformer** designed to deeply understand language by looking at **both left and right context simultaneously**.

> BERT is not a text generator — it is a **language understanding model**.

---

### 4.2 Core Features of BERT

* Encoder-only architecture
* Deeply **bidirectional context**
* Pre-training + fine-tuning paradigm
* Strong performance on understanding tasks

Model sizes:

* **BERT-base**: 12 layers, 768 hidden units, 110M parameters
* **BERT-large**: 24 layers, 1024 hidden units, 340M parameters fileciteturn1file0

---

### 4.3 BERT Pre-training Objectives

BERT is trained using **two self-supervised objectives**:

#### 1️⃣ Masked Language Modeling (MLM)

Random tokens are masked, and the model predicts them.

Example:

```
The [MASK] brown fox [MASK] over the lazy dog
```

Model predicts:

* MASK1 → *quick*
* MASK2 → *jumped*

Loss is computed **only on masked tokens**.

---

#### 2️⃣ Next Sentence Prediction (NSP)

A binary classification task:

* Does sentence B logically follow sentence A?

Example:

* A: Qasim is a good student.
* B: He studies at NUST. → **True**

This helps BERT learn **sentence-level relationships**.

---

### 4.4 Three-Pass View of BERT Training

**Pass 1 — Language Understanding**

* MLM + NSP
* Learns grammar, syntax, semantics

**Pass 2 — Representation Learning**

* Generates contextual embeddings for all tokens simultaneously

**Pass 3 — Task Adaptation**

* Fine-tuning with task-specific heads

---

### 4.5 Fine-Tuning BERT for Tasks

Common task-specific heads:

* **Text Classification** → [CLS] token + softmax
* **NER / POS** → token-level classification
* **Question Answering** → start & end span prediction

Example (QA):

* Question: What is the capital of France?
* Context: France is a country in Europe. Its capital is Paris.
* Output: Start=9, End=10 → *Paris*

---

### 4.6 Special Tokens in BERT

* `[CLS]` — classification token
* `[SEP]` — sentence separator
* `[MASK]` — masked token
* `[PAD]` — padding
* `[UNK]` — unknown token

These tokens allow BERT to support multiple NLP tasks.

---

### 4.7 BERT Variants & Optimizations

* **RoBERTa** — removes NSP, larger batches, dynamic masking
* **DistilBERT** — smaller, faster, ~95% performance
* **ALBERT** — parameter sharing, factorized embeddings
* **Longformer / BigBird** — long-context attention

Domain-specific BERTs:

* BioBERT
* SciBERT
* LegalBERT

---

## 5. GPT — Generative Pre-trained Transformer

### 5.1 What Is GPT?

**GPT** is a **decoder-only Transformer** trained to **predict the next token** given previous tokens.

> GPT is optimized for **text generation**.

---

### 5.2 GPT Training Objective

GPT uses **causal (autoregressive) language modeling**:

[ P(w_t | w_1, w_2, ..., w_{t-1}) ]

* Uses **masked self-attention**
* Looks only at past tokens

---

### 5.3 Key Properties of GPT

* Unidirectional (left-to-right)
* Excellent fluency and coherence
* Scales extremely well with data and parameters

GPT models:

* GPT-1 → GPT-2 → GPT-3 → GPT-4

---

## 6. BERT vs GPT — A Practical Comparison

| Aspect       | BERT               | GPT                   |
| ------------ | ------------------ | --------------------- |
| Architecture | Encoder-only       | Decoder-only          |
| Objective    | MLM + NSP          | Next-token prediction |
| Direction    | Bidirectional      | Unidirectional        |
| Strength     | Understanding      | Generation            |
| Typical Use  | Classification, QA | Chat, writing, coding |

Hybrid systems often use:

* BERT-style encoders for retrieval
* GPT-style decoders for generation

---

## 7. When to Use Which?

Choose **BERT** if:

* You need precise understanding
* You are doing extraction or classification

Choose **GPT** if:

* You need fluent generation
* You rely on prompting and reasoning

---

## 8. Key Takeaways

* Transformers replaced recurrence with attention
* BERT and GPT represent two complementary paradigms
* Understanding vs Generation is the core distinction
* Most modern LLM systems combine both ideas

---

##  Acknowledgement

Various contents in this presentation have been taken from different books, lecture notes, and web resources.
These materials solely belong to their respective owners and are used here only for educational clarification.
**No copyright infringement is intended.**
