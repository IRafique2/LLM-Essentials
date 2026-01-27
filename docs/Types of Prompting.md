# Day 11 — Types of Prompting: From Zero‑Shot to Multi‑Task Learning

> **Prompting is how we communicate intent to large language models.**
> The same model can behave very differently depending on *how* we ask.

---

## 1. What Is Prompting?

**Prompting** is the process of designing input instructions (text, examples, or structure) that guide a pre‑trained model to perform a specific task *without changing its parameters*.

In modern LLM workflows, prompting often **replaces or complements fine‑tuning**.

> Think of prompting as *programming with natural language*.

---

## 2. Prompting vs Fine‑Tuning

| Aspect           | Prompting            | Fine‑Tuning           |
| ---------------- | -------------------- | --------------------- |
| Model parameters | Frozen               | Updated               |
| Data requirement | None or few examples | Task‑specific dataset |
| Cost             | Low                  | High                  |
| Flexibility      | Very high            | Task‑specific         |
| Speed            | Instant              | Training required     |

Modern systems often combine both:

> **Pre‑training → Prompting → (Optional) Fine‑tuning**

---

## 3. Zero‑Shot Prompting

### Definition

**Zero‑shot prompting** means asking a model to perform a task **without providing any examples**, relying purely on its pre‑trained knowledge.

---

### How It Works

* The model has seen similar patterns during pre‑training
* A well‑written instruction activates the relevant capability

---

### Example

**Prompt:**

```
Classify the sentiment of the following sentence as Positive or Negative:
"The movie was surprisingly engaging and well‑acted."
```

**Model Output:**

```
Positive
```

---

### When to Use

* No labeled data available
* Quick prototyping
* General reasoning, summarization, Q&A

---

### Limitations

* Sensitive to wording
* Less reliable for niche tasks
* No task‑specific grounding

---

## 4. Few‑Shot Prompting

### Definition

**Few‑shot prompting** provides the model with a small number of input‑output examples (`n = 5–100`) directly inside the prompt.

> The model learns the task **from examples, not from gradient updates**.

---

### How It Works

* Demonstrations establish a pattern
* The model completes the pattern for new inputs

---

### Example

**Prompt:**

```
Text: I love this phone
Sentiment: Positive

Text: This app keeps crashing
Sentiment: Negative

Text: The service was fast and friendly
Sentiment:
```

**Model Output:**

```
Positive
```

---

### When to Use

* Limited labeled data
* Domain‑specific tasks
* Rapid adaptation

---

### Trade‑off

* Better accuracy than zero‑shot
* Prompt length grows with examples

---

## 5. Standard Fine‑Tuning (Not Prompting, but Related)

### Definition

**Standard fine‑tuning** updates model parameters using a task‑specific dataset.

---

### How It Works

1. Load a pre‑trained model
2. Use a smaller learning rate
3. Train on labeled task data
4. Optimize task‑specific loss

---

### Example Use Cases

* Sentiment classification
* Document categorization
* Medical text analysis

---

### When to Use

* Moderately sized datasets
* Stable, well‑defined tasks
* Repeated large‑scale inference

---

## 6. Zero‑Shot Learning vs Zero‑Shot Prompting

| Concept   | Zero‑Shot Learning     | Zero‑Shot Prompting   |
| --------- | ---------------------- | --------------------- |
| Field     | ML theory              | LLM usage             |
| Training  | Uses semantic labels   | No training           |
| Mechanism | Shared embedding space | Instruction following |

LLMs blur the line by enabling **zero‑shot learning through prompting**.

---

## 7. Multi‑Task Prompting / Learning

### Definition

**Multi‑task learning** trains or prompts a model to handle **multiple related tasks simultaneously**.

---

### How It Works

* Tasks share representations
* Improves generalization
* Reduces overfitting

---

### Example Prompt

```
Task 1: Summarize the text
Task 2: Identify sentiment

Text: The product arrived late but customer support resolved the issue quickly.
```

**Model Output:**

```
Summary: Delivery was delayed but support handled it well.
Sentiment: Neutral to Positive
```

---

### When to Use

* Related tasks
* Multilingual or multi‑objective systems
* General AI assistants

---

## 8. Prompting vs Parameter‑Efficient Fine‑Tuning (PEFT)

Prompting inspired modern techniques like:

* Prefix‑Tuning
* Prompt‑Tuning
* LoRA
* Adapters

These methods **learn prompts or small modules** instead of full model updates.

---

## 9. Practical Prompting Guidelines

✔ Be explicit with instructions
✔ Provide structure (lists, steps, roles)
✔ Use delimiters for clarity
✔ Show examples when accuracy matters
✔ Iterate — prompting is experimental

---

## 10. Key Takeaways

* Prompting is the primary interface to LLMs
* Zero‑shot → Few‑shot → Fine‑tuning is a spectrum
* Examples teach faster than explanations
* Multi‑task prompting improves robustness

> **Modern AI is less about training models — and more about asking the right questions.**

---

## Acknowledgement

This content is compiled from academic lectures, research papers, and educational resources and is intended solely for learning and clarification purposes.
