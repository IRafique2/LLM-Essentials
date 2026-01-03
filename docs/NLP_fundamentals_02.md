# Day 03 — NLP Layers, Tasks & Understanding Meaning

## 1. Layers of Language in NLP

Language can be thought of as having multiple layers, each representing a level of structure or meaning. NLP systems analyze these layers to extract information and make decisions.

---

### **1.1 Phonology — The Sounds of Language**

**Phonology** is the study of **sounds in a language**. It’s most important in speech-related tasks such as **Automatic Speech Recognition (ASR)**, where computers convert speech to text, and **Text-to-Speech (TTS)**, where text is converted into speech.

**Example:**

* `pin` vs `bin` → The difference of one sound completely changes the word.

> Even though phonology mainly applies to spoken language, text-based NLP sometimes uses phonological patterns, for instance, in spell-checkers or character-level models.

---

### **1.2 Morphology — How Words Are Built**

**Morphology** studies the **internal structure of words**, including roots, prefixes, suffixes, and inflections. Understanding morphology helps NLP systems recognize word meanings, handle variations, and process unseen words.

**Examples:**

* `un + happy + ness → unhappiness`
* `carried → carry + ed`
* `independently → in + depend + ent + ly`

**Applications:**

* Tokenization (splitting text into words/subwords)
* Lemmatization (reducing words to base form)
* Handling rare or new words

Modern language models like **BERT** and **GPT** use **subword tokenization** (e.g., BPE, WordPiece) to approximate morphology.

---

### **1.3 Syntax — Grammar and Sentence Structure**

**Syntax** is the study of how words are **arranged in a sentence** to form grammatically correct structures. Correct syntax ensures that a sentence is meaningful and interpretable.

**Examples:**

* *The dog bit the boy.* → A dog performed the action.
* *The boy bit the dog.* → The meaning changes completely because the word order changes.

**Applications:**

* Parsing sentences
* Part-of-speech tagging (POS)
* Phrase chunking
* Machine translation

> Modern NLP models, like Transformers, use **positional encoding** to capture some syntactic relationships.

---

### **1.4 Semantics — Understanding Literal Meaning**

**Semantics** focuses on the **literal meaning of words and sentences**. It answers the question: *What does this text actually mean?*

**Example:**

* `bank` → Could mean a river bank or a financial bank, depending on context.

**Key Semantic Tasks:**

* **Word Sense Disambiguation (WSD):** Determining which meaning of a word is intended.
* **Semantic Role Labeling (SRL):** Identifying the roles of words in a sentence, like who did what to whom.
* **Semantic Parsing:** Translating natural language into formal representations or executable commands.

**Example of SRL:**

*John drove Mary from Austin to Dallas in his Toyota Prius.*

* Agent → John
* Patient → Mary
* Source → Austin
* Destination → Dallas
* Instrument → Toyota Prius

---

### **1.5 Pragmatics — Context and Intent**

**Pragmatics** is the study of how meaning is derived from **intent and social context**. It goes beyond literal words to interpret what someone **really means**, based on the situation, relationships, and social norms.

**Example:**

> “Can you pass the salt?”
> Literally, this is a question about someone’s ability. In context, it is understood as a **polite request**.

**Pragmatics also includes:**

* Sarcasm → *“Oh great, another Monday!”* (speaker is annoyed, not happy)
* Politeness → indirect requests or softened statements
* Implicit meaning → understanding what is **implied** but not said

**Applications:** Chatbots, conversational AI, dialogue systems, sentiment analysis.

---

### **1.6 Discourse — Meaning Across Sentences**

**Discourse** studies **meaning across multiple sentences**, helping NLP systems maintain context in longer texts. It ensures that references, pronouns, and ideas are understood correctly across a passage.

**Example:**

* *Sara dropped the plate. It shattered.*
* Here, “It” refers to “the plate.”

**Applications:**

* Coreference resolution (linking pronouns to entities)
* Summarization (condensing long texts while preserving meaning)
* Long-context question answering (QA)
* Multi-turn dialogue systems

> Large language models (LLMs) like GPT learn discourse patterns through **attention mechanisms** and massive datasets, but their understanding is **data-dependent and imperfect**.

---

## 2. NLP Tasks by Layer

### **2.1 Syntactic Tasks**

* **Word Segmentation:** Splitting text into words

  ```
  jumptheshark.com → jump / the / shark / . / com
  我喜欢学习自然语言处理 → 我 / 喜欢 / 学习 / 自然语言处理
  ```

* **POS Tagging:** Assigning grammatical roles

  ```
  John saw the saw.
  John → PN, saw → V, the → Det, saw → N
  ```

* **Phrase Chunking (Shallow Parsing):** Grouping words into phrases

  ```
  [NP I] [VP ate] [NP the spaghetti] [PP with] [NP meatballs]
  ```

**Applications:** Machine translation, speech recognition, TTS, information extraction, QA

---

### **2.2 Semantic Tasks**

* **Word Sense Disambiguation (WSD)** → Identify correct meaning of words in context
* **Semantic Role Labeling (SRL)** → Detect agents, patients, locations, instruments, etc.
* **Semantic Parsing** → Convert text to logical forms or commands

**Example:**

*How many cities are there in the US?* → Logical query representation:

```
answer(A, count(B, (city(B), loc(B, C), const(C, countryid(USA))), A))
```

---

### **2.3 Pragmatics & Discourse Tasks**

* **Coreference Resolution / Anaphora:** Linking pronouns to nouns

  ```
  John put the carrot on the plate and ate it.  → it = carrot
  ```

* **Ellipsis Resolution:** Understanding omitted information

  ```
  "Wise men talk because they have something to say; fools, because they have to say something." (Plato)
  ```
---

##  High-Level NLP Applications

Understanding these semantic relations is important for many NLP applications:

* **Information Extraction:** Identify relationships between entities in text.
* **Question Answering (QA):** Use semantic relationships to infer answers.
* **Summarization:** Maintain meaning and relationships while condensing text.
* **Machine Translation:** Preserve meaning across languages.
* **Search & Autocomplete:** Suggest relevant terms based on semantic similarity.

---


---

## 3. How Computers “Understand” Language

Computers do **not truly understand language** like humans. They:

1. Convert words into **numerical representations** (embeddings)
2. Learn **patterns from large datasets**
3. Use **context and probability** to make predictions

> **Insight:** Meaning emerges from context, prior knowledge, and goals, not from words alone.

---

## 4. Lexical Resources: WordNet

**WordNet** is a **lexical database** that organizes words into **synonym sets (synsets)** and semantic relationships.

**Key Relations:**


## 1. Hypernym

*  A hypernym is a **general term or category** under which more specific words fall.
* **Example:**

  ```
  animal → dog
  vehicle → car
  ```
 >  A hypernym represents a broader concept. For instance, “animal” is a general category, and “dog” is a specific example within that category. Hypernyms help NLP systems **categorize concepts and generalize knowledge**.

---

## 2. Hyponym

* A hyponym is a **specific word that falls under a more general category** (the hypernym).
* **Example:**

  ```
  dog → puppy
  car → sedan
  ```
  > A hyponym represents a **subclass or more detailed version** of a concept. This allows NLP systems to **distinguish levels of specificity** in meaning.

---

## 3. Meronym

*  A meronym is a **word that denotes a part of something**.
* **Example:**

  ```
  wheel → car
  leaf → tree
  ```

>  Meronyms allow NLP systems to understand **part-whole relationships**, which is useful for reasoning and comprehension. For example, knowing that a wheel is part of a car helps models **connect concepts meaningfully**.

---

## 4. Antonym

* An antonym is a **word that has the opposite meaning** of another word.
* **Example:**

  ```
  hot ↔ cold
  big ↔ small
  happy ↔ sad
  ```

>   Antonyms help NLP systems **recognize contrasts and opposites**, which is useful for tasks like sentiment analysis, text understanding, and generating diverse responses.



##  Key Takeaways (Day 03)

* NLP processes language in **layers** to capture meaning at every level.
* **Phonology** → sounds
* **Morphology** → word structure
* **Syntax** → grammar and order
* **Semantics** → literal meaning
* **Pragmatics** → intent and social context
* **Discourse** → coherence across multiple sentences
* Lexical resources like **WordNet** support understanding, but **embeddings and LLMs** provide context-aware intelligence.
* Understanding these layers is critical for building **chatbots, translators, summarizers, and advanced NLP systems**.

## Acknowledgement

Various contents in this fie have been taken from different books, lecture notes, and web resources.
These materials solely belong to their respective owners and are used here only for educational and explanatory purposes.
**No copyright infringement is intended.**
