Day 09 Types of Transformer Architectures

> Not all Transformers are built the same.
Depending on the task , understanding text, generating text, or transforming one sequence into another — different Transformer architectures are used.


---

1. Why Do We Have Different Types of Transformers?

The original Transformer (Vaswani et al., 2017) introduced a full encoder–decoder architecture. However, researchers soon realized that:

Some tasks only require understanding text

Some tasks only require generation

Some tasks require both understanding and generation


This led to specialized Transformer variants, each optimized for a specific class of problems.


---

2. Encoder-Only Transformer Architecture

Definition

An Encoder-Only Transformer consists solely of a stack of encoder layers. It processes an input sequence and outputs deep contextual representations for each token.

> It does not generate text — it understands it.




---

Core Characteristics

Uses bidirectional self-attention

Each token attends to all other tokens in the input

Produces rich, context-aware embeddings

No autoregressive decoding



---

Internal Structure

Each encoder layer contains:

1. Multi-Head Self-Attention


2. Feed-Forward Neural Network


3. Residual Connections


4. Layer Normalization



Stacking multiple layers allows the model to build hierarchical semantic understanding.


---

Why Encoder-Only Models Exist

Encoder-only Transformers excel at tasks where:

The entire input is available at once

Deep understanding of context is required

No text generation is needed


They are computationally efficient for representation learning.


---

Use Cases

Text Classification

Sentiment Analysis

Named Entity Recognition (NER)

Information Retrieval

Masked Language Modeling



---

Popular Encoder-Only Models

BERT (Bidirectional Encoder Representations from Transformers)

RoBERTa

ALBERT

DeBERTa


> These models power search engines, document understanding systems, and classifiers.




---

3. Decoder-Only Transformer Architecture

Definition

A Decoder-Only Transformer consists solely of a stack of decoder layers and is designed for autoregressive text generation.

> It predicts the next token given all previous tokens.




---

Core Characteristics

Uses masked self-attention

Prevents access to future tokens

Generates text one token at a time

Fully autoregressive



---

Internal Structure

Each decoder layer contains:

1. Masked Multi-Head Self-Attention


2. Feed-Forward Neural Network


3. Residual Connections


4. Layer Normalization



There is no cross-attention, because no encoder is present.


---

Why Decoder-Only Models Exist

Decoder-only Transformers are ideal when:

The goal is text generation

The model must learn fluent language patterns

Outputs are generated sequentially


They naturally model probability distributions over text.


---

Use Cases

Language Modeling

Text Completion

Chatbots & Conversational Agents

Story & Code Generation



---

Popular Decoder-Only Models

GPT (Generative Pre-trained Transformer) series

ChatGPT

GPT-Neo / GPT-J

LLaMA


> All modern large language models are decoder-only Transformers.




---

4. Encoder–Decoder Transformer Architecture

Definition

An Encoder–Decoder Transformer uses:

An encoder to understand the input sequence

A decoder to generate the output sequence


This architecture is designed for sequence-to-sequence transformation.


---

Core Characteristics

Encoder uses bidirectional self-attention

Decoder uses:

Masked self-attention

Cross-attention over encoder outputs


Explicit alignment between input and output



---

Internal Workflow

1. Encoder converts input into contextual representations


2. Decoder attends to encoder outputs


3. Decoder generates output autoregressively



This separation allows clean division of responsibilities.


---

Why Encoder–Decoder Models Exist

They are ideal for tasks where:

Input and output sequences differ

Explicit transformation is required

Alignment between sequences matters



---

Use Cases

Machine Translation

Text Summarization

Question Answering

Text-to-Text Generation



---

Popular Encoder–Decoder Models

T5 (Text-To-Text Transfer Transformer)

BART (Bidirectional and Auto-Regressive Transformers)

Pegasus

MarianMT



---


5. Choosing the Right Transformer

Need text understanding? → Encoder-only

Need text generation? → Decoder-only

Need input → output mapping? → Encoder–Decoder


> Architecture choice should always follow the task objective.




---

6. Key Takeaways

Transformers come in three architectural flavors

Each variant is optimized for a specific class of problems

All modern NLP systems are built on these foundations


> Understanding Transformer types is essential for designing effective NLP systems.




---

Acknowledgement

This content is compiled from lecture material, research papers, and educational resources and is intended solely for learning and clarification purposes.

