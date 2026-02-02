# Day 14: Text Generation in Large Language Models (LLMs)

## 1. Introduction

Text generation is one of the most important capabilities of modern Natural Language Processing (NLP) systems. Large Language Models (LLMs) such as GPT, T5, and BART are designed to generate human-like text for a wide range of applications including chatbots, summarization, translation, and creative writing.

This article provides a detailed explanation of how text generation works, the main categories of generation models, mathematical foundations, training objectives, and evaluation methods.

---

## 2. What is Text Generation?

**Text Generation** is the task of automatically producing coherent and meaningful natural language text using a machine learning model.

The generated output may range from:

* A single token or word
* A complete sentence
* Multiple paragraphs
* Entire documents

Common examples include:

* ChatGPT responses
* Story and poem generation
* Automatic summarization
* Machine translation
* AI-based assistants

---

## 3. Goal of Text Generation

The primary goal of text generation is to produce language that resembles human writing.

A high-quality generation system should produce text that is:

* Fluent
* Coherent
* Contextually relevant
* Grammatically correct
* Logically consistent

---

## 4. Properties of Human-Like Text

A strong text generation model must satisfy three essential properties.

---

### 4.1 Coherence

**Coherence** refers to the logical flow and connectivity of ideas across sentences.

Prompt:

> Describe a beautiful sunset at the beach.

Weak response:

> The sun sets. Orange sky. Waves sound.

Coherent response:

> The sky turned into a warm palette of orange and purple as the sun slowly disappeared into the ocean, while gentle waves created a calm atmosphere.

Coherence ensures the text forms a meaningful narrative.

---

### 4.2 Fluency

**Fluency** means the text is grammatically correct and naturally written.

Fluent text avoids:

* Broken sentence structure
* Awkward phrasing
* Incorrect grammar

---

### 4.3 Relevance

**Relevance** means the generated output stays focused on the given prompt.

Prompt:

> Describe a sunset at the beach.

Irrelevant output:

> Traffic lights filled the streets of the city.

Relevant output:

> The golden sunlight reflected beautifully across the ocean waves.

---

## 5. Main Categories of Text Generation Models

Modern text generation methods can be grouped into three major categories:

1. Causal Language Modeling (CLM)
2. Sequence-to-Sequence Generation (Seq2Seq)
3. Masked Language Modeling (MLM)

---

## 6. Causal Language Modeling (CLM)

### 6.1 Definition

**Causal Language Modeling** is an autoregressive generation approach where the model predicts the next token based only on previous tokens.

Given a sequence:

$$
X = (x_1, x_2, ..., x_T)
$$

The model learns the probability:

$$
P(x_t \mid x_1, x_2, ..., x_{t-1})
$$

This means generation happens left-to-right, one token at a time.

---

### 6.2 Key Characteristics

* Autoregressive decoding
* Uses masked self-attention
* Best suited for open-ended generation

---

### 6.3 Examples of CLM Models

* GPT-2
* GPT-3
* ChatGPT
* LLaMA

---

### 6.4 Use Cases

* Chatbots
* Story generation
* Code completion
* Creative writing

---

### 6.5 Example

Prompt:

> The future of AI is

Generated continuation:

> the development of systems that can reason, learn, and assist humans.

---

## 7. Sequence-to-Sequence Generation (Seq2Seq)

### 7.1 Definition

**Sequence-to-Sequence Generation** models transform an input sequence into a different output sequence.

Formally:

$$
X = (x_1, ..., x_n)
\rightarrow
Y = (y_1, ..., y_m)
$$

The model learns:

$$
P(Y \mid X)
$$

Unlike CLM, Seq2Seq models generate text conditioned on an input.

---

### 7.2 Architecture

Seq2Seq models typically use:

* Encoder: Processes the input
* Decoder: Generates output tokens
* Cross-attention: Decoder attends to encoder representations

---

### 7.3 Examples of Seq2Seq Models

* T5
* BART
* Pegasus
* Transformer Encoder–Decoder

---

### 7.4 Use Cases

* Machine translation
* Summarization
* Question answering
* Text rewriting

---

### 7.5 Example

Input:

> The article explains climate change in detail.

Output summary:

> Climate change is driven by greenhouse gases and affects global temperatures.

---

## 8. Masked Language Modeling (MLM)

### 8.1 Definition

**Masked Language Modeling** trains a model to predict missing words using both left and right context.

Example input:

> The cat sat on the [MASK].

The objective is:

$$
P(x_{\text{mask}} \mid x_1, ..., x_{\text{mask}-1}, x_{\text{mask}+1}, ..., x_n)
$$

MLM models are bidirectional and focus more on language understanding than generation.

---

### 8.2 Examples of MLM Models

* BERT
* RoBERTa
* DistilBERT
* ALBERT

---

### 8.3 Use Cases

* Text classification
* Named Entity Recognition
* Sentiment analysis
* Information extraction

---

### 8.4 Example

Input:

> AI is transforming the [MASK] industry.

Prediction:

> healthcare

---

## 9. Text Summarization

Summarization is a core generation task where a long document is converted into a shorter version while preserving meaning.

A good summary must be:

* Fluent
* Coherent
* Informative
* Relevant

---

### 9.1 Types of Summarization

#### Single Document Summarization

One document as input, one summary as output.

#### Multi-Document Summarization

Multiple related documents combined into one summary.

---

### 9.2 Extractive vs Abstractive Summarization

#### Extractive Summarization

Selects sentences directly from the source:

$$
Summary \subset Input
$$

#### Abstractive Summarization

Generates new sentences:

$$
Summary = NewRepresentation(Input)
$$

---

## 10. Training Text Generation Models

Text generation models are trained using supervised datasets:

$$
(Text, Target)
$$

---

### 10.1 Tokenization

Raw text is converted into token IDs:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokens = tokenizer("Exercise improves health", return_tensors="pt")
```

---

### 10.2 Training Objective

Models minimize negative log-likelihood loss:


---

## 11. Inference and Generation Phase

After training, models generate text token-by-token:

```python
summary_ids = model.generate(tokens["input_ids"])
print(tokenizer.decode(summary_ids[0]))
```

Generation stops when an end token is reached.

---

## 12. Evaluation of Text Generation

### 12.1 Automatic Metrics

#### ROUGE

Measures n-gram overlap:

$$
ROUGE = \frac{\text{Overlap}}{\text{Reference}}
$$

* ROUGE-1: unigram overlap
* ROUGE-2: bigram overlap
* ROUGE-L: longest common subsequence

#### BLEU

Often used for translation tasks.

#### BERTScore

Measures semantic similarity using embeddings.

---

### 12.2 Human Evaluation

Humans evaluate generated text based on:

* Fluency
* Coherence
* Relevance
* Informativeness

---

## 13. Key Takeaways

* Text generation is central to modern LLMs
* Human-like text requires coherence, fluency, and relevance
* CLM powers GPT-style models
* Seq2Seq dominates translation and summarization
* MLM models like BERT focus on representation learning
* Evaluation uses ROUGE, BLEU, and human judgment

---

## Acknowledgment

This article is prepared for educational purposes and is inspired by a combination of:

* Lecture slides and academic course material
* Research papers on transformer-based generation models
* Publicly available NLP documentation and tutorials
* The Hugging Face Transformers library resources

All external concepts and methods belong to their respective authors and institutions. This content is intended strictly for learning and knowledge-sharing.


