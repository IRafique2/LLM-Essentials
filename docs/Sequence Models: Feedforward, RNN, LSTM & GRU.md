# Day 07 — Sequence Models: Feedforward, RNN, LSTM & GRU

Sequential data is everywhere: text, audio, time-series, stock prices, and more. Traditional feedforward neural networks cannot handle sequences effectively because they **ignore the order of inputs**. 
Recurrent architectures solve this problem by introducing **memory**.

---

## Feedforward Networks and Their Limitations

Feedforward neural networks (FNNs) are **static networks** that take a fixed-size input and produce a fixed-size output:

```
y = f(Wx + b)
```

Where:

* `x` = input vector
* `W` = weight matrix
* `b` = bias
* `f` = activation function

### Limitations for Sequential Data

1. **No memory:** Each input is independent of the previous inputs.
2. **Fixed input size:** Cannot handle variable-length sequences.
3. **No temporal relationships:** Cannot learn patterns over time or context.

**Example Problem:** Predicting the next word in a sentence.

* Feedforward network sees "I like" and "like pizza" as separate, unrelated inputs.
* Contextual meaning is lost.

 To address these issues, we use **Recurrent Neural Networks (RNNs)**.

---

## Recurrent Neural Networks (RNNs)

### Definition

A **Recurrent Neural Network** is a type of neural network designed to handle **sequential data**. RNNs maintain a **hidden state `h_t`** that acts as memory to capture information from previous time steps.

### RNN Intuition

```
h_t = f(Wx * x_t + Wh * h_{t-1} + b)
y_t = softmax(Why * h_t + by)
```

* `x_t`: input at time `t`
* `h_t`: hidden state at time `t` (memory)
* `y_t`: output at time `t`
* `Wx, Wh, Why`: learnable weight matrices

The hidden state is **updated at each time step**, allowing the network to remember previous inputs.

---

### Diagram: Vanilla RNN Cell (GitHub Markdown-friendly)

```
      x_t
       │
       ▼
   ┌─────────┐
   │  RNN    │
   │  Cell   │
   └─────────┘
       │
       ▼
      h_t
       │
       ▼
      y_t
```

* Each RNN cell receives input `x_t` and previous hidden state `h_{t-1}`.
* Produces a new hidden state `h_t` and output `y_t`.

---

### Example: Tiny Corpus

Corpus: `"I like pizza. I like cake."`
Vocabulary = {“I”, “like”, “pizza”, “cake”}

Step-by-step:

1. Input `"I"` → hidden `h1` → output predicts `"like"`
2. Input `"like"` → hidden `h2` → output predicts `"pizza"`
3. Input `"pizza"` → hidden `h3` → output predicts `"I"`

 Hidden state **remembers past context**, enabling sequence prediction.

---

### Limitations of Vanilla RNNs

* **Vanishing/Exploding Gradients:** Gradients shrink or explode over long sequences, making it hard to learn long-term dependencies.
* **Limited memory span:** Typically, only 5–10 steps can be remembered.
* **Sequential updates:** Hard to parallelize → slow training.

 **Solution:** Gated RNNs like **LSTM** and **GRU**.

---
##  Bi-directional RNNs

* Process sequences **forward and backward**
* Capture **past + future context**
* Useful for **NER, sentiment analysis, machine translation**

```
Input -> Forward RNN -> Hidden F
      -> Backward RNN -> Hidden B
Combine H = [Hidden F ; Hidden B]
```

---

##  Long Short-Term Memory (LSTM) — Detailed Discussion

**Definition:**
LSTM is a type of RNN designed to **remember long-term information** and mitigate the vanishing gradient problem. Introduced by Hochreiter & Schmidhuber (1997), LSTM has **gates** that control the flow of information.

### LSTM Components
| Component         | Purpose                                               |
| ----------------- | ----------------------------------------------------- |
| Cell state (c_t)  | Stores long-term memory                               |
| Hidden state(h_t) | Stores short-term output                              |
| Forget gate       | Decides what info to discard from previous cell state |
| Input gate        | Decides what new info to write to the cell state      |
| Output gate       | Decides what part of cell state to output             |

1. **Cell state (`c_t`)**

   * Think of it as a **conveyor belt** running through the sequence.
   * It carries **long-term memory** with minimal modification.

2. **Hidden state (`h_t`)**

   * The **short-term memory** that is exposed to the next layer or output.
   * Carries only the information deemed relevant by the output gate.

3. **Forget Gate (`f_t`)**

   * Decides **what part of previous memory to discard**.
   * Sigmoid output between 0 and 1; 1 = keep all, 0 = forget all.
   * Equation:

     ```
     f_t = σ(Wf·[h_{t-1}, x_t] + bf)
     ```

4. **Input Gate (`i_t`)**

   * Controls **how much new information to write** to the cell.
   * Works with the **candidate memory `ĉ_t`**.
   * Equation:

     ```
     i_t = σ(Wi·[h_{t-1}, x_t] + bi)
     ĉ_t = tanh(Wc·[h_{t-1}, x_t] + bc)
     ```

5. **Output Gate (`o_t`)**

   * Determines **which part of the cell state to output** as hidden state.
   * Equation:

     ```
     o_t = σ(Wo·[h_{t-1}, x_t] + bo)
     h_t = o_t ⊙ tanh(c_t)
     ```

---

### LSTM Step-by-Step Intuition

Imagine you are **reading a story** and need to remember key events:

1. **Forget gate:** Decides which past events are irrelevant. (“Do I care about the character from chapter 1?”)
2. **Input gate:** Adds **new, important information** to memory. (“A new character enters — remember this!”)
3. **Output gate:** Reveals what is currently **relevant** for the next prediction. (“For the next sentence, only remember the plot twists, not every detail.”)

 This structure allows LSTMs to **accumulate knowledge over long sequences** without overwriting everything at each step.

---

### Diagram: LSTM Cell 

```
![LSTM Cell](../images/lstm.png)

---

### PyTorch Example: LSTM

```python
import torch
import torch.nn as nn

# Sample sequence: batch_size=1, seq_len=3, input_size=2
x = torch.tensor([[[1,2],[3,4],[5,6]]], dtype=torch.float)

lstm = nn.LSTM(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
output, (h_n, c_n) = lstm(x)

print("Output shape:", output.shape)  # (batch, seq_len, hidden_size)
print("Hidden state:", h_n.shape)
print("Cell state:", c_n.shape)
```

---

##  Gated Recurrent Unit (GRU) — Detailed Discussion

**Definition:**
GRU is a simplified LSTM, introduced by Cho et al., which **combines cell and hidden states** into one and uses **two gates**: **update** and **reset**.

### GRU Components

1. **Update Gate (`z_t`)**

   * Decides **how much of the previous hidden state to keep**.
   * Similar to combining forget + input gates in LSTM.

2. **Reset Gate (`r_t`)**

   * Controls **how much past information to forget** when calculating candidate hidden state.

3. **Candidate hidden state (`ĥ_t`)**

   * The new memory calculated using the reset gate.

4. **Hidden state (`h_t`)**

   * Final memory for the current step: combines previous state and candidate based on the update gate.

---

### GRU Equations

```
z_t = σ(Wz·x_t + Uz·h_{t-1})      # update gate
r_t = σ(Wr·x_t + Ur·h_{t-1})      # reset gate
ĥ_t = tanh(W·x_t + U·(r_t ⊙ h_{t-1})) # candidate hidden
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ ĥ_t # final hidden state
```

---

### GRU Step-by-Step Intuition

Think of GRU as a **simpler LSTM**:

* **Reset gate:** “Forget some past information before combining with new input.”
* **Update gate:** “Decide how much old memory to retain vs how much new info to accept.”

 GRUs are **lighter, faster to train**, and often work just as well as LSTMs for many tasks.

---

### PyTorch Example: GRU

```python
gru = nn.GRU(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
output, h_n = gru(x)

print("Output shape:", output.shape)
print("Hidden state:", h_n.shape)
```

---



## KeyTakeaways

* **Feedforward:** No memory, fixed-size input.

* **RNN:** Adds hidden state → remembers recent past, but struggles with long sequences.

* **LSTM:** Introduces gates + cell state → remembers long-term dependencies.

* **GRU:** Simplified LSTM → fewer gates, combined memory, faster training.

* All are **sequential models** but **transformers have now replaced them in most NLP tasks**.

---
## Acknowledgement
Parts of this material are adapted from books, lecture notes, and online educational resources.
All rights belong to their respective owners.
Used strictly for educational purposes.
---
