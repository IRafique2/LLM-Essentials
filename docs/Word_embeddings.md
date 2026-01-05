#  Day 05 — Word Embeddings 

Day 05 — Word Embeddings

Language models cannot work directly with words.
Before machines can understand meaning, words must be converted into numbers—but not just any numbers.

This is where word embeddings come in.

Why Do We Need Word Embeddings?

Humans understand that:

cat and dog are similar

king and queen are related

apple (fruit) and Apple (company) depend on context

For a computer, words are just symbols.

Traditional representations like one-hot encoding fail because:

They are extremely sparse

They encode no similarity

Every word is equally distant from every other word

📌 Key Insight:
To capture meaning, words must be represented in a continuous vector space, where distance encodes similarity.

What Are Word Embeddings?

Word embeddings are dense vector representations of words where:

Each word is mapped to a vector (e.g., 100–300 dimensions)

Words used in similar contexts have similar vectors

Semantic and syntactic relationships emerge naturally

Definition:

Word embeddings encode meaning by learning from contextual co-occurrence, not predefined rules.

Intuition: Personality Embeddings Analogy

Think of a personality test. You might score:

Introversion ↔ Extroversion

Logical ↔ Emotional

Risk-averse ↔ Risk-seeking

Each person becomes a point in multi-dimensional space.

👉 Word embeddings work the same way — but the dimensions are learned automatically, not labeled.

Semantic Geometry: The Famous Example

Word embeddings allow vector arithmetic:

king − man + woman ≈ queen


This works because embeddings learn latent dimensions such as:

Gender

Royalty

Age

Profession

📌 This is not magic — it is geometry in vector space.

🔗 Visual intuition:
https://jalammar.github.io/illustrated-word2vec/

🔍 Types of Word Embeddings

Over time, NLP has evolved through three major embedding paradigms:

Frequency-based embeddings

Prediction-based embeddings

Contextual embeddings

Each paradigm builds on the previous one, improving semantic understanding.

## What Is Word2Vec?

**Word2Vec** is a technique for learning word embeddings using a **shallow neural network**.

Key ideas:

* Learn from raw text
* No labeled data required
* Context defines meaning

📌 **Important:**
Word2Vec does **not** store explicit features like *gender* or *royalty*.
Those relationships **emerge from data**.

---

## Word2Vec Architecture (High Level)

Word2Vec is a **2-layer neural network**:

1. **Input layer**

   * Words represented as one-hot vectors
2. **Hidden layer**

   * Dense embedding layer (this is what we care about)
3. **Output layer**

   * Predicts words or context

At the end of training:
👉 **The hidden layer weights become the word embeddings**

🔗 Architecture visual:
[https://mccormickml.com/assets/word2vec/word2vec_skipgram_net_arch.png](https://mccormickml.com/assets/word2vec/word2vec_skipgram_net_arch.png)

---

## Two Word2Vec Training Approaches

Word2Vec can be trained in **two ways**:

* **CBOW (Continuous Bag of Words)**
* **Skip-Gram**

Both learn embeddings — they just ask **opposite questions**.

---

## CBOW — Predict the Target Word

**CBOW predicts a word using its surrounding context.**

### Example Sentence

> *“The cake was chocolate flavoured”*

CBOW task:

```
Context → Target
(The, was, chocolate, flavoured) → cake
```

### How CBOW Works

1. Convert context words to one-hot vectors
2. Map them to embeddings
3. Average embeddings
4. Predict the target word using softmax

📌 **Properties**

* Faster to train
* Works well with frequent words
* Smooths information across context

🔗 CBOW illustrated:
[https://jalammar.github.io/images/word2vec/cbow.png](https://jalammar.github.io/images/word2vec/cbow.png)

---

## Skip-Gram — Predict the Context

**Skip-Gram predicts surrounding words given a center word.**

Using the same sentence:

```
Input → Outputs
cake → The, was, chocolate, flavoured
```

### Why Skip-Gram Is Powerful

* Each (center, context) pair is a training example
* Better at learning rare words
* Works well with smaller datasets

📌 **Trade-off:** Slower, but more expressive

🔗 Skip-Gram illustrated:
[https://jalammar.github.io/images/word2vec/skipgram.png](https://jalammar.github.io/images/word2vec/skipgram.png)

---

## Window Size: Controlling Context

The **window size** determines how many neighboring words are considered.

Example (window = 2):

```
The quick brown fox jumps over the lazy dog
                ↑
Context: quick, brown, jumps, over
```

* Small window → syntactic relationships
* Large window → semantic relationships

---

## Training Skip-Gram: The Core Idea

For each training step:

1. Take one **positive pair**
   (word, true neighbor)
2. Add several **negative samples**
   (word, random non-neighbors)
3. Optimize embeddings so:

   * Positive pairs get higher similarity
   * Negative pairs get lower similarity

📌 This converts softmax classification into **logistic regression**, making training efficient.

---

## Negative Sampling (Why It Matters)

Without negative samples:

* Model could predict “1” for everything
* Learn nothing useful

Negative sampling forces the model to **discriminate**:

> “Which words *do not* belong in this context?”

🔗 Negative sampling intuition:
[https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

---

## What Happens After Training?

After training completes:

* Discard the output/context matrix
* Keep the embedding matrix
* Each row = vector for one word

These embeddings can now be reused for:

* Sentiment analysis
* Text classification
* Question answering
* Machine translation
* As inputs to LLMs

---

## Limitations of Word2Vec

Despite its impact, Word2Vec has limits:

* One embedding per word (no context awareness)
* Cannot handle polysemy well
  (*bank* = river vs finance)
* Static embeddings
* Cannot model long-range context

📌 These limitations led to **contextual embeddings** (ELMo, BERT, GPT).

---

## Why Word2Vec Still Matters (Even in LLM Era)

Word2Vec:

* Introduced distributed semantic representations
* Proved meaning can emerge from context
* Inspired all modern embedding techniques

👉 **LLMs are not replacements — they are evolutions.**

---

## ✅ Key Takeaway (Day 05)

* Words must be converted into vectors to model meaning
* Word embeddings capture similarity geometrically
* Word2Vec learns embeddings from context
* CBOW and Skip-Gram are two sides of the same idea
* This is the bridge from **statistical NLP → neural NLP → LLMs**

---

## 📚 Further Reading & Visual References

* Illustrated Word2Vec (Highly Recommended)
  [https://jalammar.github.io/illustrated-word2vec/](https://jalammar.github.io/illustrated-word2vec/)

* The Inner Workings of Word2Vec
  [https://mccormickml.com/2019/03/12/the-inner-workings-of-word2vec/](https://mccormickml.com/2019/03/12/the-inner-workings-of-word2vec/)

* Dummy’s Guide to Word2Vec
  [https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673](https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673)

* Word2Vec Explained (Academic)
  [https://www.cambridge.org/core/journals/natural-language-engineering/article/word2vec/B84AE4446BD47F48847B4904F0B36E0B](https://www.cambridge.org/core/journals/natural-language-engineering/article/word2vec/B84AE4446BD47F48847B4904F0B36E0B)

---

## 📌 Acknowledgement

Various contents in this presentation have been taken from different books, lecture notes, and web resources.
These materials solely belong to their respective owners and are used here only for educational clarification.
**No copyright infringement is intended.**

---

If you want next, I can:

* 🔜 **Day 06 — Neural Language Models (RNNs & LSTMs)**
* 🔜 Add **code examples (NumPy / PyTorch)**
* 🔜 Create **LinkedIn + Twitter summaries**
* 🔜 Design a **learning roadmap graphic**

Just say **Day 06** 🚀
