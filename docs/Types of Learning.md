# Day 11 — Types of Learning: From Classical ML to Modern LLM Training

> **Learning paradigms define *how* a machine learns, *what supervision it receives*, and *how knowledge is transferred*.**

---

## 1. What Do We Mean by "Learning"?

In Machine Learning, **learning** refers to the process of adjusting model parameters so that:

* Predictions improve over time
* Error is minimized
* Patterns are extracted from data

Mathematically, learning means minimizing a **loss function**:

```
θ* = argmin_θ 𝓛(y, f(x; θ))
```

Where:

* `x` = input data
* `y` = target (if available)
* `θ` = model parameters
* `𝓛` = loss function

---

## 2. Supervised Learning

### Definition

**Supervised Learning** uses **fully labeled data**, where each input has a corresponding output label.

```
(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)
```

### How It Works

* Model predicts `ŷ`
* Loss is computed: `𝓛(y, ŷ)`
* Parameters updated using gradient descent

### Example

* Email spam classification
* Image classification

### Mathematical Example (Solved)

Given:

```
y = 2x + 1
x = [1, 2]
y = [3, 5]
```

Prediction:

```
ŷ = wx + b
```

Loss (MSE):

```
𝓛 = (1/n) Σ (y - ŷ)²
```

Gradient descent updates `w, b` to minimize error.

---

## 3. Unsupervised Learning

### Definition

**Unsupervised Learning** discovers patterns from **unlabeled data**.

### Key Goals

* Clustering
* Density estimation
* Dimensionality reduction

### Example

* Customer segmentation
* Topic modeling

### Mathematical Example (K-Means)

Distance metric:

```
||x - μ_k||²
```

Cluster assignment:

```
k = argmin ||x - μ_k||
```

Centroid update:

```
μ_k = (1/N) Σ x_i
```

---

## 4. Semi-Supervised Learning

### Definition

**Semi-Supervised Learning** uses:

* A small labeled dataset
* A large unlabeled dataset

```
Labeled + Unlabeled → Better Generalization
```

### Why It Exists

* Labels are expensive
* Data is abundant

### Example

* Labeling 100 medical reports
* Using them to infer labels for 10,000 reports

### Mathematical Idea

Total loss:

```
𝓛 = 𝓛_supervised + λ 𝓛_unsupervised
```

Where:

* `λ` controls unlabeled influence

---

## 5. Reinforcement Learning (RL)

### Definition

**Reinforcement Learning** learns by **interaction with an environment** using rewards.

### Core Components

* Agent
* Environment
* State (s)
* Action (a)
* Reward (r)

### Objective

Maximize expected cumulative reward:

```
R = Σ γᵗ r_t
```

### Example

* Robot navigation
* Game playing

### Solved Math Example

Given rewards:

```
[+1, -1, +2]
γ = 0.9
```

Return:

```
R = 1 + 0.9(-1) + 0.9²(2) = 1.72
```

---

## 6. Challenges with Supervised Learning

* Requires large labeled datasets
* Expensive annotation
* Training from scratch is costly

### Solution → Transfer Learning

---

## 7. Transfer Learning

### Definition

**Transfer Learning** reuses knowledge from a **source task** to improve a **target task**.

```
Source Domain → Target Domain
```

### Why It Works

* Lower data requirements
* Faster convergence
* Better generalization

### Example

* ImageNet → Medical X-rays
* BERT → Sentiment analysis

---

## 8. Types of Transfer Learning

### 8.1 Domain Adaptation

* Same task
* Different domains

**Example:**
Movie reviews → Product reviews

---

### 8.2 Inductive Transfer Learning

* Different tasks
* Related domains

**Example:**
English–French → English–Spanish translation

---

### 8.3 Transductive Transfer Learning

* Same task
* Different data distributions

**Example:**
News sentiment → Social media sentiment

---

## 9. Pre-Training

### Definition

Training on **large, generic datasets** to learn universal patterns.

### Examples

* BERT (masked language modeling)
* GPT (next-token prediction)
* ImageNet (vision)

---

## 10. Feature Extraction

Using frozen representations from pre-trained models:

```
Input → Pre-trained Model → Features → Classifier
```

Benefits:

* Reduced training time
* Less data required

---

## 11. Fine-Tuning

### Definition

Fine-tuning adapts a pre-trained model using a **small task-specific dataset**.

### Mathematical View

```
θ_new = θ_pretrained + Δθ
```

Only small updates are made to parameters.

---

## 12. Training vs Pre-Training vs Fine-Tuning

| Concept           | Purpose                   |
| ----------------- | ------------------------- |
| Training          | Learn a task from scratch |
| Pre-training      | Learn general knowledge   |
| Transfer Learning | Reuse learned knowledge   |
| Fine-tuning       | Specialize for a task     |

---

## 13. Learning Pipeline in LLMs

```
Unsupervised / Semi-supervised Pre-training
        ↓
Supervised Fine-Tuning
        ↓
Task-Specific LLM
```

---

## 14. Key Takeaways

* Learning type defines supervision level
* Transfer learning enables modern AI scale
* LLMs combine multiple learning paradigms

> **Modern AI is not trained — it is transferred, adapted, and refined.**

---

## Acknowledgement

This content is compiled from academic lectures, research literature, and learning resources for educational purposes only.
