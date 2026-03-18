import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

CSV_PATH = "diabetes_sample_5000.csv"
TARGET_COL = "diabetes_label"


def add_clinical_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    g0  = X.get("glucose_fasting")
    g1  = X.get("ogtt_1h_glucose")
    g2  = X.get("ogtt_2h_glucose")
    bmi = X.get("bmi")
    ins = X.get("insulin_fasting")
    cpe = X.get("c_peptide_fasting")
    age = X.get("age")

    if g0 is not None and g1 is not None:
        X["ogtt_delta_1h"]   = (g1 - g0).astype(float)
        X["ogtt_slope_0_1h"] = (g1 - g0).astype(float)
    if g0 is not None and g2 is not None:
        X["ogtt_delta_2h"]   = (g2 - g0).astype(float)
        X["ogtt_slope_0_2h"] = (g2 - g0).astype(float)
    if g1 is not None and g2 is not None:
        X["ogtt_slope_1_2h"] = (g2 - g1).astype(float)
    if g0 is not None and g1 is not None and g2 is not None:
        X["ogtt_auc_trap"] = (0.5*(g0 + g1) + 0.5*(g1 + g2)).astype(float)

    if ins is not None and g0 is not None:
        X["homa_ir"] = (ins * g0 / 405.0).astype(float)

    if g0 is not None:
        X["fpg_prediabetes"] = ((g0 >= 100) & (g0 < 126)).astype(int)
        X["fpg_diabetes"]    = (g0 >= 126).astype(int)
    if g2 is not None:
        X["ogtt2h_prediabetes"] = ((g2 >= 140) & (g2 < 200)).astype(int)
        X["ogtt2h_diabetes"]    = (g2 >= 200).astype(int)
    if bmi is not None:
        X["bmi_overweight"] = ((bmi >= 25) & (bmi < 30)).astype(int)
        X["bmi_obese"]      = (bmi >= 30).astype(int)

    if ins is not None and cpe is not None:
        X["insulin_cpep_ratio"] = (ins / (cpe + 1e-6)).astype(float)
    if bmi is not None and g0 is not None:
        X["bmi_x_fpg"] = (bmi * g0).astype(float)
    if bmi is not None and age is not None:
        X["bmi_x_age"] = (bmi * age).astype(float)

    for col in ["age", "bmi", "glucose_fasting"]:
        if col in X.columns:
            X[f"{col}_sq"] = X[col].astype(float) ** 2

    for col in ["insulin_fasting", "c_peptide_fasting", "glucose_fasting"]:
        if col in X.columns:
            X[f"log1p_{col}"] = np.log1p(X[col].astype(float).clip(lower=0))

    return X


@st.cache_resource(show_spinner="Training model...")
def load_model():
    if not Path(CSV_PATH).exists():
        return None, None, None

    df = pd.read_csv(CSV_PATH)
    df["gender"] = (df["gender"].str.strip().str.lower() == "male").astype(int)

    ID_LIKE = {"id", "patient_id"}
    feature_cols = [
        c for c in df.columns
        if c != TARGET_COL and c.lower() not in ID_LIKE and not c.lower().endswith("id")
    ]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)

    # pick threshold from validation set
    proba_val = pipe.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    J = prec[:-1] + rec[:-1] - 1.0
    threshold = float(thr[int(np.nanargmax(J))]) if thr.size > 0 else 0.5

    return pipe, feature_cols, threshold


def build_input() -> pd.DataFrame:
    st.sidebar.header("Patient readings")

    age    = st.sidebar.slider("Age (years)", 18, 90, 52)
    gender = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True)
    bmi    = st.sidebar.slider("BMI (kg/m²)", 15.0, 55.0, 29.4, step=0.1)

    st.sidebar.markdown("**Fasting labs**")
    fpg  = st.sidebar.slider("Fasting glucose (mg/dL)", 60, 400, 118)
    ins  = st.sidebar.slider("Fasting insulin (µU/mL)", 1.0, 200.0, 12.3, step=0.1)
    cpep = st.sidebar.slider("C-peptide (ng/mL)", 0.1, 15.0, 1.8, step=0.1)

    st.sidebar.markdown("**OGTT**")
    ogtt1 = st.sidebar.slider("OGTT 1h glucose (mg/dL)", 60, 500, 165)
    ogtt2 = st.sidebar.slider("OGTT 2h glucose (mg/dL)", 60, 500, 185)

    return pd.DataFrame([{
        "age": age,
        "gender": 1 if gender == "Male" else 0,
        "bmi": bmi,
        "glucose_fasting": fpg,
        "insulin_fasting": ins,
        "c_peptide_fasting": cpep,
        "ogtt_1h_glucose": ogtt1,
        "ogtt_2h_glucose": ogtt2,
    }])


def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="", layout="wide")

    st.title("Diabetes Risk Predictor")
    st.caption("Logistic Regression trained on 5,000 synthetic patient records. For educational purposes only — not a medical device.")

    model, feature_cols, threshold = load_model()

    if model is None:
        st.error(f"Dataset not found: `{CSV_PATH}`. Make sure the CSV is in the same directory as app.py.")
        return

    X_raw = build_input()
    X_fe = add_clinical_features(X_raw)

    # align columns to what the model was trained on
    for col in feature_cols:
        if col not in X_fe.columns:
            X_fe[col] = 0.0
    X_fe = X_fe[feature_cols]

    proba = float(model.predict_proba(X_fe)[0, 1])
    pred  = int(proba >= threshold)

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability", f"{proba:.1%}")
    col2.metric("Threshold", f"{threshold:.2f}")
    col3.metric("Prediction", "AT RISK" if pred == 1 else "LOW RISK")

    if pred == 1:
        st.error("**AT RISK** — model predicts elevated diabetes risk based on the values provided.")
    else:
        st.success("**LOW RISK** — model does not flag elevated diabetes risk based on the values provided.")

    with st.expander("Engineered features passed to model"):
        st.dataframe(X_fe.T.rename(columns={0: "value"}), use_container_width=True)

    st.markdown("---")
    st.caption("Dataset is synthetic. Results are illustrative only and should not guide clinical decisions.")


if __name__ == "__main__":
    main()
