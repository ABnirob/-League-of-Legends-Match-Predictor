# League of Legends Match Predictor

A logistic regression model, built with **PyTorch**, that predicts the
outcome of ranked League of Legends matches from early-game (10-minute)
team statistics. Built for the "Final Project: League of Legends Match
Predictor" assignment.

## 📁 Project Structure

```
lol_project/
├── README.md                     <- You are here
├── requirements.txt               <- Python dependencies
├── data/
│   ├── generate_dataset.py        <- Script that generates the dataset
│   ├── build_notebook.py          <- Script that programmatically builds the .ipynb
│   └── league_of_legends_data_stats.csv
├── notebook/
│   ├── Final_Project_League_of_Legends_Match_Predictor.ipynb  <- MAIN DELIVERABLE (Q1 upload)
│   └── league_of_legends_data_stats.csv  <- copy, so the notebook runs standalone
├── models/
│   └── logistic_regression_model.pth     <- Saved trained model weights
├── images/
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── lr_tuning.png
│   └── feature_importance.png
└── answers/
    └── Assignment_Answers.md      <- Answers/rationale for quiz Questions 2–10
```

## 🎯 Objective

Predict `blueWins` (1 = Blue team wins, 0 = Blue team loses) using team-level
statistics captured at the 10-minute mark of the match (kills, deaths,
assists, gold, experience, wards, dragons, heralds, towers, CS, etc.).

## 🗂️ Dataset

`league_of_legends_data_stats.csv` — 9,879 ranked matches × 37 columns
(36 features + `blueWins` target), modeled on the structure of the
well-known "High Diamond Ranked 10-Min" League of Legends dataset (Blue vs.
Red team early-game stats). Generated via `data/generate_dataset.py` with a
fixed random seed (42) so results are fully reproducible. Class balance is
~50/50 (blueWins: 49.3% / 50.7%).

> If your course provides an official dataset file, simply replace
> `league_of_legends_data_stats.csv` in both `data/` and `notebook/` with the
> official file (same column names expected) and re-run the notebook.

## 🧠 Model

A single-layer logistic regression model implemented in PyTorch:

```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
```

- **Loss:** Binary Cross-Entropy (`nn.BCELoss`)
- **Optimizer:** Adam, with `weight_decay=0.01` (L2 regularization)
- **Epochs:** 1000
- **Preprocessing:** `StandardScaler` on all 36 features, 80/20 stratified train/test split

## 📊 Results (this run)

| Metric | Value |
|---|---|
| Train Accuracy | 81.63% |
| Test Accuracy | 79.96% |
| Precision | 0.7990 |
| Recall | 0.7924 |
| F1-score | 0.7957 |
| ROC AUC | 0.8935 |
| Best learning rate (tuning) | 0.001 (80.06% test acc.) |
| Top 3 features | redTotalGold, redGoldPerMin, blueTotalGold |

## ▶️ How to Reproduce

```bash
# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Regenerate the dataset
cd data && python generate_dataset.py && cd ..

# 3. Run the notebook
jupyter notebook notebook/Final_Project_League_of_Legends_Match_Predictor.ipynb
# or, to re-execute headlessly:
jupyter nbconvert --to notebook --execute --inplace \
  notebook/Final_Project_League_of_Legends_Match_Predictor.ipynb
```

## 📝 Assignment Submission Checklist

- [x] **Q1:** Upload `notebook/Final_Project_League_of_Legends_Match_Predictor.ipynb` (fully executed, all 8 sub-parts present)
- [x] **Q2–Q10:** See `answers/Assignment_Answers.md` for answers and reasoning based on this notebook's actual output

## 📦 Dependencies

See `requirements.txt`. Core libraries: `torch`, `pandas`, `numpy`,
`scikit-learn`, `matplotlib`.
