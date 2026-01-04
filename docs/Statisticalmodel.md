#  Day 04 — Statistical Language Models & N-Grams

Modern Large Language Models may look magical, but at their core lies a very simple idea:
 **predicting what word comes next**.

Before neural networks and transformers, this idea was formalized using **statistical language models**. Understanding these models is crucial because they form the **conceptual foundation of today’s LLMs**.

---

## What Is a Language Model?

A **language model (LM)** assigns probabilities to sequences of words.

There are two closely related goals:

### 1. Sentence Probability

How likely is this entire sentence?

```
P(w1, w2, w3, ..., wn)
```

### 2. Next-Word Prediction

Given previous words, what word comes next?

```
P(wn | w1, w2, ..., wn-1)
```

Any model that can compute either of these is called a **Language Model**.

---

## Why Language Models Matter

Language models are everywhere, often invisibly:

* **Speech Recognition**

  * *“I bought fresh mangoes from the market”*
    is far more likely than
    *“I bot fresh man goes from the mar kit”*

* **Machine Translation**

  * *“Heavy rainfall”* sounds better than *“Big rainfall”*
  * *“Festival of lights”* sounds more natural than *“Festival of lamps”*

* **Spelling Correction**

* **Text Generation**

* **Autocomplete & Search**

In all these cases, the system selects the **most probable word sequence**.

---

## From Sentences to Probabilities: The Chain Rule

Consider the sentence:

> **“The monsoon season has begun”**

Using the **Chain Rule of Probability**:

```
P(w1, w2, ..., wn) = P(w1) × P(w2 | w1) × P(w3 | w1, w2) × ... × P(wn | w1, ..., wn-1)
```

For our sentence:

```
P(The) × P(monsoon | The) × P(season | The monsoon) × P(has | The monsoon season) × ...
```

### The Problem

* Contexts grow very long
* Exact word sequences are rare
* Data becomes sparse quickly

This makes probability estimation unreliable.

---

## The Markov Assumption: Simplifying the Problem

To make language modeling practical, we apply the **Markov Assumption**:

> The next word depends only on the **previous k words**, not the full history.

This allows models to focus on **local context**.

---

## N-Gram Language Models

An **N-gram model** predicts a word using only the previous **N−1 words**.

### Common Types

* **Unigram**

```
P(begun)
```

* **Bigram**

```
P(begun | has)
```

* **Trigram**

```
P(begun | season has)
```

📌 **Key Insight:**
An **N-gram language model** is an **(N−1)-order Markov model**.

---

## Estimating Probabilities with MLE

The simplest estimation method is **Maximum Likelihood Estimation (MLE)**.

### Bigram Probability

```
P(wi | wi-1) = count(wi-1, wi) / count(wi-1)
```

This relies entirely on **relative frequencies** from training data.

---

## The Data Sparsity Problem

MLE assumes that test data resembles training data.

### Example

**Training Data**

* enjoyed the movie
* enjoyed the food
* enjoyed the game

**Test Data**

* enjoyed the concert
* enjoyed the festival

Since these bigrams never appeared:

```
P(concert | enjoyed the) = 0
```

This leads to:

* Zero-probability sentences
* Broken perplexity
* Poor generalization

---

## Smoothing: Fixing Zero Probabilities

To handle unseen word sequences, we use **smoothing**.

### Laplace (Add-One) Smoothing

Idea:

> Pretend every word appeared **once more** than it actually did.

```
P(wi | wi-1) =(count(wi-1, wi) + 1) / (count(wi-1) + |V|)
```

* Prevents zero probabilities
*  Can over-smooth for large vocabularies

---

### Add-k Smoothing

Instead of adding 1, add a smaller constant:

```
k = 0.1 or 0.01
```

This produces more balanced probability estimates.

---

### Unigram Prior Smoothing

* Use unigram probabilities as prior knowledge
* Frequent words get higher pseudo-counts
* More linguistically realistic

---

## Back-off and Interpolation

As **N increases**, context improves but sparsity worsens.

### Back-off

* Try trigram
* If unseen → use bigram
* If unseen → use unigram

### Interpolation

Combine all models:

```
P = λ1 × Puni + λ2 × Pbi + λ3 × Ptri
```

 **Interpolation usually outperforms back-off.**

---

## Limitations of N-Gram Models

* Cannot model long-range dependencies
* Fixed context window
* Large memory requirements
* Vocabulary explosion

 Example:

> *“The project, which he had been working on for months, was finally approved.”*
> The dependency between *project* and *approved* is too distant.

---

## Why N-Grams Still Matter

Despite their limitations:

* Introduced probabilistic language modeling
* Formalized next-word prediction
* Inspired neural language models

 **LLMs are essentially scaled, neural versions of this same idea.**

---

##  Key Takeaway (Day 04)

* Language models predict word sequences
* N-grams use probability + Markov assumptions
* Smoothing is essential for real-world data
* N-grams laid the foundation for modern LLMs

---

##  Acknowledgement

Various contents in this presentation have been taken from different books, lecture notes, and web resources.
These materials solely belong to their respective owners and are used here only for educational clarification.
**No copyright infringement is intended.**


