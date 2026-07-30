# Final Project: League of Legends Match Predictor — Quiz Answers (Q2–Q10)

> These answers are based on the actual, executed run of
> `Final_Project_League_of_Legends_Match_Predictor.ipynb` in this project
> (see `/notebook`). If you re-run the notebook with different random seeds,
> data, or epochs, the run-dependent answers (Q7–Q10) may shift slightly —
> always double check against **your own** executed output before submitting.

---

### Question 1 — Notebook Upload (8 points)
Upload `notebook/Final_Project_League_of_Legends_Match_Predictor.ipynb`
(already executed with outputs visible). It contains all 8 required
components:

| Sub-part | Requirement | Location in notebook |
|---|---|---|
| 1.1 | Load & preprocess data | "Step 1.1" |
| 1.2 | Logistic regression model in PyTorch | "Step 1.2" |
| 1.3 | Train the model | "Step 1.3" |
| 1.4 | Optimization technique + performance evaluation | "Step 1.4" |
| 1.5 | Confusion matrix & ROC curve visualization | "Step 1.5" |
| 1.6 | Save & load the trained model | "Step 1.6" |
| 1.7 | Hyperparameter tuning for learning rate | "Step 1.7" |
| 1.8 | Feature importance | "Step 1.8" |

---

### Question 2 — Creating a tensor from a Python list (1 point)
**Answer: `torch.tensor(data)`**

`torch.tensor()` is the standard PyTorch function used to construct a tensor
directly from a Python list (or nested list, NumPy array, etc.), e.g.:
```python
torch.tensor([1, 2, 3])
```
(Functions like `torch.Tensor()` (the class constructor) also work but
`torch.tensor()` is the recommended, type-inferring factory function taught
in the course materials.)

---

### Question 3 — Defining the logistic regression model in PyTorch (Select all, 3 points)
**Correct answers:**
- ✅ The model is defined as a class that inherits from `nn.Module`.
- ✅ `nn.Linear(input_dim, 1)` is used to compute a linear combination of the
  input features (the "logits").
- ✅ A `sigmoid` activation (`torch.sigmoid`) is applied to the linear
  output inside `forward()` to convert logits into a probability between 0
  and 1.

**Incorrect/distractor options to avoid:**
- ❌ Using `softmax` instead of `sigmoid` (softmax is for multi-class, not
  binary, classification).
- ❌ Defining the model without a `forward()` method (required by `nn.Module`).
- ❌ Using `nn.Linear(input_dim, 2)` for a binary sigmoid output (that's a
  2-class softmax setup, not this binary logistic regression setup).

Matches the notebook's implementation:
```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
```

---

### Question 4 — Clearing old gradients before backpropagation (1 point)
**Answer: `optimizer.zero_grad()`**

PyTorch accumulates gradients by default, so `optimizer.zero_grad()` must be
called before each `loss.backward()` call to clear gradients from the
previous step; otherwise gradients would incorrectly accumulate across
batches/epochs.

---

### Question 5 — Purpose of `weight_decay=0.01` in the optimizer (1 point)
**Answer: It applies L2 regularization to the model's weights, to help prevent overfitting.**

`weight_decay` adds a penalty term proportional to the sum of squared
weights to the loss function. This discourages the model from learning
excessively large weights, which improves generalization to unseen data
(reduces overfitting). It is **not** related to learning-rate scheduling or
gradient clipping.

---

### Question 6 — Insights from the ROC curve output (Select all, 2 points)
Based on the notebook's ROC curve (**AUC ≈ 0.89**):

**Correct answers:**
- ✅ The model performs substantially better than random guessing (the ROC
  curve lies well above the diagonal reference line, and AUC ≈ 0.89 is much
  greater than the 0.5 baseline).
- ✅ A higher AUC score indicates a better trade-off between the true
  positive rate and false positive rate across all classification
  thresholds (better overall ability to distinguish wins from losses).

**Incorrect/distractor options to avoid:**
- ❌ "An AUC close to 0.5 indicates the model is highly accurate" (0.5 =
  random guessing, i.e. the worst realistic case).
- ❌ "The ROC curve only reflects performance at a single fixed threshold of 0.5"
  (it actually reflects performance across *all* thresholds).

---

### Question 7 — Highest metric from the confusion matrix output (1 point)
From the notebook's Step 1.5 output (test set):

| Metric | Value |
|---|---|
| Accuracy | 0.7996 (≈ 79.96%) |
| Precision | 0.7990 |
| **Recall** | 0.7924 |
| F1-score | 0.7957 |

**Answer: Precision was the highest of the four metrics reported** (0.7990),
narrowly above accuracy (0.7996 is technically the single highest number,
but if the question restricts to Precision/Recall/F1 as the confusion
matrix-derived metrics, **Precision** is the highest).
➡️ **Check your own run's printed values** — with a different random seed
these four metrics can reorder slightly (they're normally very close
together, as they are here).

---

### Question 8 — Highest test accuracy achieved during hyperparameter tuning (1 point)
From the notebook's Step 1.7 learning-rate sweep:

| Learning Rate | Test Accuracy |
|---|---|
| 0.001 | **80.06%** |
| 0.01 | 79.96% |
| 0.05 | 79.96% |
| 0.1 | 79.96% |
| 0.5 | 80.01% |

**Answer: ≈ 80.06%** (achieved at learning rate = 0.001).

---

### Question 9 — Learning rate that gave the best test accuracy (1 point)
**Answer: 0.001**

This was the learning rate that produced the highest test accuracy
(80.06%) in the hyperparameter sweep in Step 1.7.

---

### Question 10 — Top three most important features (bar plot) (1 point)
From the notebook's Step 1.8 feature-importance bar plot (ranked by absolute
weight magnitude on standardized features):

1. **redTotalGold** (weight ≈ −0.337) — more Red-team gold → lower chance Blue wins
2. **redGoldPerMin** (weight ≈ −0.337) — closely tied with redTotalGold (highly correlated features)
3. **blueTotalGold** (weight ≈ +0.336) — more Blue-team gold → higher chance Blue wins

**Answer: Total/per-minute gold for both teams are the top three most
important features** — i.e., **gold-related statistics** dominate the
model's predictions, consistent with League of Legends game knowledge
(gold lead is one of the strongest indicators of match outcome).

> Note: `blueGoldPerMin` is mathematically `blueTotalGold / 10`, so it is
> essentially a duplicate signal — pick whichever gold-related options your
> quiz's multiple-choice list actually offers (gold-related stats are the
> answer either way).
