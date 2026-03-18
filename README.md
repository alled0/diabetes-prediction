# Diabetes Risk Prediction

Predicts diabetes risk from routine lab values and OGTT readings. Compares Logistic Regression, Random Forest, XGBoost, and a Keras neural net on a 5,000-patient synthetic dataset.

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

> Replace the link above after deploying to [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## Results

| Model | AUROC | Avg Precision | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 | 0.999 |
| XGBoost | 1.000 | 1.000 | 1.000 | 0.996 |
| Neural Net (Keras) | 1.000 | 1.000 | 0.999 | 0.997 |
| Logistic Regression | 0.958 | 0.947 | 0.872 | 0.846 |

Threshold selected per-model by maximising the F1 point on the PR curve (validation set).

---

## Project structure

```
diabetes-prediction/
├── diabetes_prediction.ipynb   # main notebook
├── diabetes_sample_5000.csv    # dataset
├── app.py                      # Streamlit demo
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

## Run the notebook

```bash
jupyter lab
```

Open `diabetes_prediction.ipynb` and run cells top-to-bottom:

1. Load and encode data
2. Feature engineering (`add_clinical_features`)
3. Train / validate / test split
4. Train all models, compare metrics
5. Use the interactive prediction form at the bottom

## Run the Streamlit app locally

```bash
streamlit run app.py
```

## Features engineered

From the raw labs the notebook derives:
- OGTT slopes and deltas (0→1h, 1→2h, 0→2h)
- Trapezoidal AUC of the glucose curve
- HOMA-IR (fasting insulin × fasting glucose / 405)
- Clinical threshold flags (pre-diabetes / diabetes by FPG, OGTT-2h, BMI)
- Log-transforms of skewed lab values
- Interaction terms (BMI × FPG, BMI × age)

## Notes

- Dataset is synthetic and for educational purposes only.
- Not a medical device. Do not use for clinical decisions.
