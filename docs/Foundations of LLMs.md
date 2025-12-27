# Day 01 – Foundations of LLMs

## How Programming Started

Early programming relied on **rule-based systems**, where a specific rule had to be defined for every possible situation.  
- This approach was **rigid**, hard to adapt to complex real-world scenarios, and **prone to errors**.

---

## Artificial Intelligence (AI)

With the limitations of traditional programming, **Artificial Intelligence (AI)** emerged.  
- AI aims to **mimic human brain functionality** to perform intelligent tasks.  
- Computers are programmed to learn, reason, and make decisions.

---

## Machine Learning (ML)

**Machine Learning (ML)** shifted the paradigm:  
- Instead of explicit rules, computers **learn patterns from data** and make decisions based on approximations.  
- ML is broadly categorized into:
  - **Supervised Learning** – learns from labeled data  
  - **Unsupervised Learning** – finds patterns in unlabeled data  
  - **Semi-Supervised Learning** – uses a combination of labeled and unlabeled data  

> ML works best with **structured data**.

---

## Deep Learning and Artificial Neural Networks

To handle **unstructured data** such as images and text, Deep Learning (DL) emerged as a subfield of Machine Learning. Deep Learning models are inspired by the structure and function of the **human brain**, using interconnected units called **neurons** to learn complex patterns from data.

### What is an Artificial Neural Network (ANN)?

An **Artificial Neural Network (ANN)** is a computing model that mimics the brain’s network of neurons to process information and recognize patterns. In an ANN:
- Each input is multiplied by a **weight** that reflects its importance.
- A **bias** term is added.
- An **activation function** determines the output of a neuron.

The basic computation for a neuron in a multilayer perceptron (MLP) is:

\[
a = f\Big(\sum_{i=1}^{n} w_i x_i + b \Big)
\]

Where:
- \(x_i\) = input features  
- \(w_i\) = weights  
- \(b\) = bias  
- \(f(\cdot)\) = activation function  
- \(a\) = neuron's output

This process allows the network to learn complex, nonlinear relationships in data.

### How ANNs Work

A neural network consists of:
- **Input layer** – receives raw data
- **Hidden layer(s)** – intermediate neurons that learn internal representations
- **Output layer** – produces the final prediction

During training:
1. Inputs are forwarded through the network.
2. Predictions are compared against ground truth.
3. Errors are propagated backward (backpropagation) to update weights.

This learning mechanism enables the network to improve performance over time.

### Source & Diagram Reference

For a clear visual explanation of how an ANN structure and its connections work, see the diagram in *“How does Artificial Neural Network (ANN) algorithm work? Simplified!”* on Analytics Vidhya:  
https://www.analyticsvidhya.com/blog/2014/10/ann-work-simplified/ :contentReference[oaicite:0]{index=0}

> **Note:** Neural networks are inspired by biological neural systems, but they are much simpler mathematically and operate with weighted sums and activation functions to learn from data. :contentReference[oaicite:1]{index=1}


---

## Large Language Models (LLMs)

**Large Language Models (LLMs)** are deep learning models trained on **massive datasets with billions of parameters**:  
-Large refer to the amount of data and number of paramerters (weights and biasness) model was trained on.
- They learn patterns in data and can **generate synthetic content**.  
- LLMs can be broadly divided into:

The Evolution from traditional AI/ML/DL workflows to generative LLMs was shown below.
![Evolution from traditional AI/ML/DL workflows to generative LLM](images/aivsmlvsds.png)

---

### 1. Generative Models
- Generate new content **similar to training data**, but not exact copies.  
- Example: Text generation, image generation.

### 2. Discriminative Models
- Learn the **boundary between classes** for prediction or classification.  
- Example: Spam detection, sentiment analysis.
   
The diagram below shows a comparison between **traditional programming/AI approaches** and **generative models**, illustrating the evolution to Large Language Models (LLMs):

![Comparison of Traditional Programming with Generative Models](images/tdvsds.png)

---

## Key Takeaways

- Programming evolved from **rule-based systems → AI → ML → DL → LLMs**.  
- ML focuses on **structured data**, DL enables **unstructured data processing**.  
- LLMs combine **deep learning + large-scale data** to create generative and discriminative AI systems.


