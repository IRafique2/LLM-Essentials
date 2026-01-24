#  Day 06 — Sequence Modeling & Decoding Strategies

---

## Sequence Modeling:
**Definition:**
Sequence modeling is a machine learning approach where the **order of data points matters**, and predictions depend on **previous elements in the sequence**.

In NLP, sequence modeling means:

> Predicting the next word based on all previously generated words.

**Mathematical Representation:**

$$
P(w_t \mid w_1, w_2, \dots, w_{t-1})
$$

This expresses:

> The probability of the current word given all past words.

---

###  Example:

**Sentence:**

```
"I love deep learning"
```

At token `"learning"`, the model learns:

$$
P(\text{learning} \mid \text{I, love, deep})
$$

If `"deep"` came after `"love"`, the probability of `"learning"` **increases**.

 Word order is critical.

---

️## What Is Decoding? 

**Definition:**
Decoding is the process of converting a model’s **probability distribution over vocabulary** into **actual words** during text generation.

At each timestep:

1. Model outputs logits
2. Apply softmax → probabilities
3. Decoding strategy selects the next token

---

###  Example:

**Model Output Probabilities**

| Word     | Probability |
| -------- | ----------- |
| "of"     | 0.50        |
| "in"     | 0.25        |
| "with"   | 0.15        |
| "banana" | 0.10        |

 Decoding decides **which word is chosen**.

---

️### Why Decoding Strategies Matter

**Definition:**
Decoding strategies control how **confident, diverse, or creative** the generated text is.

They influence:

* Fluency
* Repetition
* Creativity
* Randomness vs determinism

---

###  Example

**Prompt:** `"The future of AI is …"`

| Strategy | Behavior            |
| -------- | ------------------- |
| Greedy   | Safe & repetitive   |
| Sampling | Creative & diverse  |
| Beam     | Formal & structured |

---

##  Greedy Decoding

**Definition:**
Greedy decoding selects the word with the **highest probability** at each step.

$$
w_t = \arg\max P(w)
$$

---

### Example 

**Prompt:** `"Paris is the city"`

**Step 1:** Next word probabilities

| Token     | Probability |
| --------- | ----------- |
| "of"      | 0.4         |
| "in"      | 0.25        |
| "with"    | 0.15        |
| "history" | 0.1         |
| "banana"  | 0.05        |
| "future"  | 0.05        |

* Max probability = `"of"`
  ➡ Sentence: `"Paris is the city of"`

**Step 2:** Next word probabilities (after `"of"`)

| Token     | Probability |
| --------- | ----------- |
| "history" | 0.35        |
| "culture" | 0.3         |
| "future"  | 0.2         |
| "banana"  | 0.05        |
| "of"      | 0.1         |

* Max probability = `"history"`
  ➡ Final Output: `"Paris is the city of history"`

---

**Advantages:**

*  Fast
* Simple
* Low computation

**Disadvantages:**

* Repetitive loops
*  No creativity
*  Local optimum only

 Deterministic — same input always gives same output

---

## Beam Search

**Definition:**
Beam Search keeps the **top-k most probable sequences** at each timestep instead of only one.

---

### Example 

**Step 1:** Initial expansion (top-2 tokens)

| Token | Score |
| ----- | ----- |
| "of"  | 0.4   |
| "in"  | 0.25  |

**Step 2:** Expand each sequence

**Sequence 1:** `"of"`

| Next Word | Probability | Cumulative Score  |
| --------- | ----------- | ----------------- |
| history   | 0.35        | 0.4 × 0.35 = 0.14 |
| culture   | 0.3         | 0.12              |
| future    | 0.2         | 0.08              |

**Sequence 2:** `"in"`

| Next Word | Probability | Cumulative Score   |
| --------- | ----------- | ------------------ |
| future    | 0.3         | 0.25 × 0.3 = 0.075 |
| culture   | 0.25        | 0.0625             |

**Step 3:** Pick top 2 sequences by cumulative score

* `"of history"` → 0.14 
* `"of culture"` → 0.12 

**Final Output:**

```
"Paris is the city of history"
```

---

**Advantages:**

* Better sentence quality
*  Less greedy decisions

**Disadvantages:**

* Computationally expensive
*  Still deterministic
*  Can repeat phrases

---

##  Sampling-Based Decoding

**Definition:**
Sampling selects the next token **randomly based on probabilities**, instead of always picking the max.

---

### Example

| Word   | Probability |
| ------ | ----------- |
| "of"   | 0.5         |
| "in"   | 0.3         |
| "with" | 0.2         |

**Sampled outputs may be:**

* `"of"` → most common
* `"in"` or `"with"` → occasionally

 Adds **variation and creativity**

---

️## Temperature Sampling 

**Definition:**
Temperature $T$ controls the **sharpness of the probability distribution**:

$$
P(w) = \frac{e^{z/T}}{\sum e^{z/T}}
$$

* Low T → high probability words dominate
* High T → rare words more likely

---

### Example

| Temperature | Effect                  |
| ----------- | ----------------------- |
| T = 0.1     | Almost greedy           |
| T = 1.0     | Normal randomness       |
| T = 1.5     | Highly creative/unusual |

---

## Top-K Sampling

**Definition:**
Keep only the **K most probable tokens**, then sample from them.

### Example

| Word | Probability |
| ---- | ----------- |
| A    | 0.4         |
| B    | 0.3         |
| C    | 0.2         |
| D    | 0.1         |

* K = 2 → Keep {A, B}
* Sample only from these → prevents rare words

---

## Top-P (Nucleus) Sampling

**Definition:**
Keep the **smallest set of tokens** whose **cumulative probability ≥ p**.

### Example (p = 0.9)

| Word | Probability | Cumulative |
| ---- | ----------- | ---------- |
| A    | 0.4         | 0.4        |
| B    | 0.3         | 0.7        |
| C    | 0.2         | 0.9       |
| D    | 0.1         | 1.0        |

* Keep {A, B, C}
* Sample from them → adaptive diversity

---



##  Key Takeaways

* Sequence modeling captures **context & order**
* Models output **probabilities, not words**
* Decoding decides **how text is formed**
* Greedy & Beam → **accuracy-focused**
* Sampling & Top-P → **creativity-focused**
* Modern LLMs often combine **Temperature + Top-P** for best results

---
Acknowledgement

Various contents in this material have been adapted from lecture slides, textbooks, and online resources.
All rights belong to their respective owners and are used strictly for educational purposes.
No copyright infringement is intended.
