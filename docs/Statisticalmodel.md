#  Day 04 — Statistical Language Models & N-Grams

Modern Large Language Models may look magical, but at their core lies a very simple idea:
 **predicting what word comes next**.

Before neural networks and transformers, this idea was formalized using **statistical language models**. Understanding these models is crucial because they form the **conceptual foundation of today’s LLMs**.

---

## What Is a Language Model?

A **language model (LM)** assigns probabilities to sequences of words.

There are two closely related goals:

1. **Sentence Probability**

   > How likely is this entire sentence?

   [
   P(w_1, w_2, w_3, \dots, w_n)
   ]

2. **Next-Word Prediction**

   > Given previous words, what word comes next?

   [
   P(w_n \mid w_1, w_2, \dots, w_{n-1})
   ]

Any model that can compute either of these is called a **Language Model**.

---

## Why Language Models Matter

Language models are everywhere, often invisibly:

* **Speech Recognition**

  * *“I bought fresh mangoes from the market”*
    is far more likely than
    *“I bot fresh man goes from the mar kit”*

* **Machine Translation**

  * *“Heavy rainfall”* is preferred over *“Big rainfall”*
  * *“Festival of lights”* sounds more natural than *“Festival of lamps”*

* **Spelling Correction**

* **Text Generation**

* **Autocomplete & Search**

In all these cases, the model chooses the **most probable sequence of words**.

---

## From Sentences to Probabilities: The Chain Rule

Consider the sentence:

> **“The monsoon season has begun”**

To compute its probability, we apply the **Chain Rule of Probability**:

[
P(w_1, w_2, \dots, w_n) = P(w_1) \cdot P(w_2 \mid w_1) \cdot \dots \cdot P(w_n \mid w_1, \dots, w_{n-1})
]

So this becomes:

[
P(\text{The}) \times P(\text{monsoon} \mid \text{The}) \times P(\text{season} \mid \text{The monsoon}) \times \dots
]

### The Problem

In real language:

* Contexts get very long
* Exact word sequences are rare
* Data quickly becomes sparse

This makes estimating probabilities unreliable.

---

## The Markov Assumption: Simplifying the Problem

To make language modeling practical, we introduce the **Markov Assumption**:

> The next word depends only on the **previous k words**, not the entire history.

This approximation allows us to ignore distant context and focus on recent words.

---

## N-Gram Language Models

An **N-gram model** predicts a word using only the previous **N-1 words**.

### Common Types

* **Unigram**
  [
  P(\text{begun})
  ]

* **Bigram**
  [
  P(\text{begun} \mid \text{has})
  ]

* **Trigram**
  [
  P(\text{begun} \mid \text{season has})
  ]

📌 **Key Insight**
An **N-gram language model** is equivalent to an **(N−1)-order Markov model**.

---

## Estimating Probabilities with MLE

The simplest way to estimate N-gram probabilities is **Maximum Likelihood Estimation (MLE)**.

### Bigram Example

[
P(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})}
]

This uses **relative frequencies** from the training data.

---

## The Data Sparsity Problem

MLE works well **only if test data looks like training data**.

Consider:

**Training Data**

* *enjoyed the movie*
* *enjoyed the food*
* *enjoyed the game*

**Test Data**

* *enjoyed the concert*
* *enjoyed the festival*

Since these bigrams never appeared during training:

[
P(\text{concert} \mid \text{enjoyed the}) = 0
]

This causes:

* Zero probability sentences
* Undefined perplexity
* Poor generalization

---

## Smoothing: Fixing Zero Probabilities

To handle unseen N-grams, we use **smoothing techniques**.

### Laplace (Add-One) Smoothing

The idea:

> Pretend we saw every word **one extra time**.

[
P_{\text{Add-1}}(w_i \mid w_{i-1}) =
\frac{c(w_{i-1}, w_i) + 1}{c(w_{i-1}) + |V|}
]

This ensures:

* No zero probabilities
* Better robustness

⚠️ But Add-1 can be **too aggressive**, especially for large vocabularies.

---

### Add-k Smoothing

Instead of adding 1, add a smaller constant *k* (e.g., 0.1 or 0.01).

This provides a softer correction.

---

### Unigram Prior Smoothing

Instead of adding equal mass to all words:

* Use unigram probabilities as a **prior**
* Common words receive higher pseudo-counts

This approach is more realistic and flexible.

---

## Back-off and Interpolation

As **N increases**, context improves but data sparsity worsens.

### Back-off

* Use trigram if available
* Otherwise fall back to bigram
* Otherwise unigram

### Interpolation

* Combine all models:
  [
  P = \lambda_1 P_{\text{uni}} + \lambda_2 P_{\text{bi}} + \lambda_3 P_{\text{tri}}
  ]

📌 **Interpolation usually performs better** than pure back-off.

---

## Limitations of N-Gram Models

Despite their usefulness, N-gram models have serious limitations:

* Cannot capture **long-range dependencies**
* Context window is fixed and small
* Vocabulary grows rapidly
* Memory and computation scale poorly

📌 Example:

> *“The project, which he had been working on for months, was finally approved.”*
> The dependency between *project* and *approved* is too far apart for N-grams.

---

## Why N-Grams Still Matter

Even though modern LLMs use neural networks:

* N-grams introduced **probabilistic language modeling**
* They formalized **next-word prediction**
* Many ideas (context, smoothing, back-off) inspired neural approaches

👉 **LLMs are powerful generalizations of this same idea**, trained at massive scale.

---

## ✅ Key Takeaway (Day 04)

* Language modeling is about **predicting word sequences**
* N-gram models use probability, the chain rule, and Markov assumptions
* Smoothing is essential to handle unseen data
* N-grams laid the groundwork for modern Large Language Models

---

## 📌 Acknowledgement

Various contents in this presentation have been taken from different books, lecture notes, and web resources.
These materials solely belong to their respective owners and are used here only for educational and explanatory purposes.
**No copyright infringement is intended.**


Just say **“Day 05”** 🚀
