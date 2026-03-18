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

LIFESTYLE_WEIGHTS = {
    "family_history": 0.12,
    "hypertension":   0.10,
    "sedentary":      0.09,
    "high_sugar_diet": 0.08,
    "smoking":        0.06,
}


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not Path(CSV_PATH).exists():
        return None, None

    df = pd.read_csv(CSV_PATH)
    df["gender"] = (df["gender"].str.strip().str.lower() == "male").astype(int)

    features = ["age", "gender", "bmi"]
    X = df[features].copy()
    y = df[TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)

    proba_val = pipe.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    J = prec[:-1] + rec[:-1] - 1.0
    threshold = float(thr[int(np.nanargmax(J))]) if thr.size > 0 else 0.5

    return pipe, threshold


def lifestyle_adjustment(answers: dict) -> float:
    total = sum(LIFESTYLE_WEIGHTS[k] for k, v in answers.items() if v)
    return min(total, 0.40)


def risk_label(score: float):
    if score < 0.25:
        return "Low Risk", "#27ae60", "Your responses suggest a low risk of diabetes."
    if score < 0.50:
        return "Moderate Risk", "#e67e22", "Some risk factors are present. Consider discussing with a doctor."
    return "High Risk", "#c0392b", "Several risk factors are present. A medical check-up is recommended."


def main():
    st.set_page_config(page_title="Diabetes Risk Check", page_icon="", layout="centered")

    st.title("Diabetes Risk Check")
    st.caption("Answer a few simple questions to get a personal risk estimate. No lab tests needed.")
    st.markdown("---")

    model, threshold = load_model()
    if model is None:
        st.error(f"Data file not found: `{CSV_PATH}`.")
        return

    # --- Section 1: Basic info ---
    st.subheader("About you")
    col1, col2 = st.columns(2)
    age    = col1.slider("How old are you?", 18, 90, 40)
    gender = col2.radio("Gender", ["Male", "Female"], horizontal=True)

    col3, col4 = st.columns(2)
    height_cm = col3.number_input("Height (cm)", min_value=120, max_value=230, value=170)
    weight_kg = col4.number_input("Weight (kg)", min_value=30, max_value=250, value=75)

    bmi = weight_kg / ((height_cm / 100) ** 2)
    bmi_category = (
        "Underweight" if bmi < 18.5 else
        "Normal"      if bmi < 25   else
        "Overweight"  if bmi < 30   else
        "Obese"
    )
    st.caption(f"Your BMI: **{bmi:.1f}** ({bmi_category})")

    st.markdown("---")

    # --- Section 2: Lifestyle ---
    st.subheader("Lifestyle & health history")

    family_history = st.checkbox("A parent or sibling has been diagnosed with diabetes")
    hypertension   = st.checkbox("I have been told I have high blood pressure")
    smoking        = st.checkbox("I smoke or have smoked in the past 5 years")

    activity = st.select_slider(
        "How physically active are you?",
        options=["Rarely / never", "Light (1-2x per week)", "Moderate (3-4x per week)", "Very active (daily)"],
        value="Light (1-2x per week)",
    )
    sedentary = activity == "Rarely / never"

    diet = st.select_slider(
        "How would you describe your daily diet?",
        options=["Lots of sugary / processed food", "Mixed", "Mostly whole foods / vegetables"],
        value="Mixed",
    )
    high_sugar_diet = diet == "Lots of sugary / processed food"

    st.markdown("---")

    # --- Compute risk ---
    X_input = pd.DataFrame([{
        "age":    age,
        "gender": 1 if gender == "Male" else 0,
        "bmi":    bmi,
    }])
    base_proba = float(model.predict_proba(X_input)[0, 1])

    lifestyle_adj = lifestyle_adjustment({
        "family_history":  family_history,
        "hypertension":    hypertension,
        "smoking":         smoking,
        "sedentary":       sedentary,
        "high_sugar_diet": high_sugar_diet,
    })

    final_score = min(base_proba + lifestyle_adj, 0.97)
    label, color, advice = risk_label(final_score)

    # --- Result ---
    st.subheader("Your result")

    st.markdown(
        f"""
        <div style="border: 2px solid {color}; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: {color};">{label}</div>
            <div style="font-size: 1.1rem; margin-top: 8px; color: #555;">{advice}</div>
            <div style="font-size: 0.9rem; margin-top: 12px; color: #888;">Risk score: {final_score:.0%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(final_score, 1.0))

    # --- Breakdown ---
    with st.expander("What's driving your score?"):
        factors = []
        if bmi >= 30:
            factors.append(("BMI in obese range", "high"))
        elif bmi >= 25:
            factors.append(("BMI in overweight range", "medium"))
        if age >= 45:
            factors.append(("Age 45 or older", "medium"))
        if family_history:
            factors.append(("Family history of diabetes", "high"))
        if hypertension:
            factors.append(("High blood pressure", "medium"))
        if smoking:
            factors.append(("Smoking history", "medium"))
        if sedentary:
            factors.append(("Low physical activity", "medium"))
        if high_sugar_diet:
            factors.append(("High sugar / processed diet", "medium"))

        if not factors:
            st.write("No major risk factors detected in your answers.")
        else:
            for name, level in factors:
                icon = "🔴" if level == "high" else "🟡"
                st.write(f"{icon} {name}")

    st.markdown("---")
    st.caption(
        "This tool is for educational purposes only and is not a medical diagnosis. "
        "If you are concerned about your diabetes risk, please consult a healthcare professional."
    )


if __name__ == "__main__":
    main()
