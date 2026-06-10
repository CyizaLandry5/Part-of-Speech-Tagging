# 🧠 Assignment 2: Parts-of-Speech Tagging with Hidden Markov Models

**Course:** Natural Language Processing Specialization (Course 2)  
**Topics:** POS Tagging, Hidden Markov Models (HMM), Viterbi Algorithm

---

## 📌 Project Overview

This project implements a **Part-of-Speech (POS) Tagger** using both a simple frequency-based approach and a more sophisticated **Hidden Markov Model (HMM)** with the **Viterbi algorithm**.

POS tagging is the task of labeling each word in a sentence with its grammatical role (noun, verb, adjective, etc.). This is challenging because many words are **ambiguous** — they can represent different parts of speech depending on context.

**Example of ambiguity with the word "well":**

| Sentence | POS Tag |
|----------|---------|
| The team played **well**. | Adverb |
| You are doing **well** for yourself. | Adjective |
| **Well**, this took forever. | Interjection |
| The **well** is dry. | Noun |
| Tears began to **well** up. | Verb |

---

## 🎯 What I Built

### 1. Frequency-Based Baseline Tagger
- Assigns the most frequent POS tag for each word based on training data
- Accuracy: **92.53%**

### 2. Hidden Markov Model (HMM) Tagger
- Uses transition probabilities (tag → next tag) and emission probabilities (tag → word)
- Implements the **Viterbi algorithm** for dynamic programming decoding
- Accuracy: **95.31%**

---

## 📊 Dataset

| Dataset | Purpose | Source |
|---------|---------|--------|
| `WSJ_02-21.pos` | Training | Wall Street Journal |
| `WSJ_24.pos` | Testing | Wall Street Journal |
| `hmm_vocab.txt` | Vocabulary (words appearing ≥2 times + unknown tokens) | Processed from training |

- **46 unique POS tags** (e.g., NN, VB, JJ, RB, IN, DT, etc.)
- ~86% of tokens are unambiguous (only one possible tag)
- ~14% are ambiguous (multiple possible tags)

---

## 🛠️ Implementation Details

### Core Functions Implemented

| Function | Purpose |
|----------|---------|
| `create_dictionaries()` | Builds transition counts, emission counts, and tag counts from training corpus |
| `predict_pos()` | Baseline: predicts most frequent tag for each word |
| `create_transition_matrix()` | Computes smoothed transition probabilities A(tag | prev_tag) |
| `create_emission_matrix()` | Computes smoothed emission probabilities P(word | tag) |
| `initialize()` | Initializes Viterbi matrices with start state |
| `viterbi_forward()` | Dynamic programming forward pass |
| `viterbi_backward()` | Backtracks to find optimal tag sequence |
| `compute_accuracy()` | Evaluates predictions against ground truth |

### Smoothing Formula Used

To avoid zero probabilities:

**Transition smoothing:**  
`P(t_i | t_{i-1}) = (C(t_{i-1}, t_i) + α) / (C(t_{i-1}) + α × N)`

**Emission smoothing:**  
`P(w_i | t_i) = (C(t_i, w_i) + α) / (C(t_i) + α × V)`

Where:
- `α = 0.001` (smoothing parameter)
- `N` = number of POS tags (46)
- `V` = vocabulary size

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Baseline (most frequent tag) | 92.53% |
| HMM + Viterbi | **95.31%** |

The HMM outperforms the baseline by **~2.8%** by leveraging sequential context rather than treating each word independently.

---

---

## ❓ Reflection Questions

These questions helped me think deeper about the assignment:

### On Ambiguity
> *Why does the baseline tagger achieve ~92% accuracy even though ~14% of tokens are ambiguous?*

The baseline works well because even though 14% of word types are ambiguous, many of those ambiguous words still have one **dominant** tag that appears much more frequently than others. So for most occurrences, guessing the most common tag is correct.

### On HMM Advantages
> *Why does the HMM improve accuracy over the baseline?*

The HMM considers the **previous tag's identity**, not just the word itself. For ambiguous words (e.g., "back" can be RB, VB, NN, JJ, RP, VBP), the transition probability helps disambiguate — if the previous word is "to" (TO), "back" is likely a verb (VB); if previous is "came" (VBD), "back" is likely an adverb (RB).

### On Smoothing
> *Why is smoothing necessary in both A and B matrices?*

Without smoothing, any unseen (tag, tag) or (tag, word) pair would have zero probability, causing the entire Viterbi path probability to become zero (or -inf in log space). This would fail on any test word not seen in training with a particular tag.

### On Log Probabilities
> *Why do we use log probabilities instead of raw probabilities?*

Raw probabilities become extremely small (many < 10^-50), causing numerical underflow in floating-point arithmetic. Log probabilities convert multiplication to addition, maintaining numerical stability while preserving the argmax relationship.

### On Complexity
> *What is the time complexity of the Viterbi algorithm, and why?*

O(T × N²), where T = number of words (≈30,000) and N = number of tags (46). For each word, we compute N × N transitions from all previous tags to all current tags. This is feasible because N is small (46), but would be problematic with a larger tagset.

### On Limitations
> *What are the limitations of this first-order HMM?*

- Only considers **previous tag**, not longer-range dependencies
- Cannot handle tags that depend on **future** words (bidirectional context)
- Assumes each word depends only on its own tag (naive emission independence)

---

## 🚀 How to Run

```bash
# Run all cells in the Jupyter notebook
# The notebook will:
# 1. Load WSJ training and test data
# 2. Build transition and emission matrices
# 3. Run baseline prediction
# 4. Run Viterbi HMM prediction
# 5. Output accuracy scores
## 🔍 Sample Predictions
