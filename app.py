import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = "model.joblib"
FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def age_to_brfss(age):
    for i, upper in enumerate([24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79]):
        if age <= upper:
            return i + 1
    return 13


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not Path(MODEL_PATH).exists():
        return None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["threshold"]


def risk_level(pred, proba):
    if pred == 0:
        return "Low Risk", "#27ae60", "No major risk factors detected."
    if proba < 0.65:
        return "Moderate Risk", "#e67e22", "Some risk factors present. Small lifestyle changes can make a big difference."
    return "High Risk", "#c0392b", "Several risk factors are present. A check-up with your doctor is a good idea."


def main():
    st.set_page_config(page_title="Diabetes Risk Check", layout="centered")

    st.title("Diabetes Risk Check")
    st.caption(
        "Based on the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 — "
        "253,680 real survey responses. For educational purposes only."
    )
    st.markdown("---")

    model, threshold = load_model()
    if model is None:
        st.error(f"Model file not found: `{MODEL_PATH}`. Run the notebook save cell first.")
        return
    feature_cols = FEATURE_COLS

    # --- Section 1: About you ---
    st.subheader("About you")
    col1, col2 = st.columns(2)
    age = col1.slider("Age", 18, 90, 45)
    sex = col2.radio("Sex", ["Female", "Male"], horizontal=True)

    col3, col4 = st.columns(2)
    height_cm = col3.number_input("Height (cm)", 120, 230, 170)
    weight_kg = col4.number_input("Weight (kg)", 30, 250, 75)

    bmi = weight_kg / (height_cm / 100) ** 2
    bmi_label = (
        "Underweight" if bmi < 18.5 else
        "Normal"      if bmi < 25   else
        "Overweight"  if bmi < 30   else
        "Obese"
    )
    st.caption(f"BMI: **{bmi:.1f}** — {bmi_label}")

    education = st.selectbox(
        "Highest level of education",
        options=[
            ("Never attended school / kindergarten", 1),
            ("Elementary school (grades 1–8)", 2),
            ("Some high school (grades 9–11)", 3),
            ("High school graduate", 4),
            ("Some college or technical school", 5),
            ("College graduate", 6),
        ],
        format_func=lambda x: x[0],
        index=3,
    )
    income = st.selectbox(
        "Annual household income",
        options=[
            ("Less than $10,000", 1),
            ("$10,000 to $15,000", 2),
            ("$15,000 to $20,000", 3),
            ("$20,000 to $25,000", 4),
            ("$25,000 to $35,000", 5),
            ("$35,000 to $50,000", 6),
            ("$50,000 to $75,000", 7),
            ("More than $75,000", 8),
        ],
        format_func=lambda x: x[0],
        index=5,
    )

    st.markdown("---")

    # --- Section 2: Health history ---
    st.subheader("Health history")
    col5, col6 = st.columns(2)
    high_bp    = col5.checkbox("I have high blood pressure")
    high_chol  = col6.checkbox("I have high cholesterol")
    chol_check = col5.checkbox("Cholesterol checked in past 5 years", value=True)
    stroke     = col6.checkbox("I have had a stroke")
    heart      = col5.checkbox("I have heart disease or have had a heart attack")
    diff_walk  = col6.checkbox("I have difficulty walking or climbing stairs")

    st.markdown("---")

    # --- Section 3: Lifestyle ---
    st.subheader("Lifestyle")
    col7, col8 = st.columns(2)
    smoker    = col7.checkbox("I have smoked at least 100 cigarettes in my life")
    phys_act  = col8.checkbox("I was physically active in the past 30 days", value=True)
    fruits    = col7.checkbox("I eat fruit 1 or more times per day", value=True)
    veggies   = col8.checkbox("I eat vegetables 1 or more times per day", value=True)
    alcohol   = col7.checkbox("I drink heavily (men: 14+ drinks/week, women: 7+ drinks/week)")

    st.markdown("---")

    # --- Section 4: General health ---
    st.subheader("How have you been feeling?")
    gen_hlth = st.select_slider(
        "Overall, how would you rate your general health?",
        options=["Excellent", "Very Good", "Good", "Fair", "Poor"],
        value="Good",
    )
    gen_hlth_val = ["Excellent", "Very Good", "Good", "Fair", "Poor"].index(gen_hlth) + 1

    col9, col10 = st.columns(2)
    ment_hlth = col9.slider("Days of poor mental health in the past 30 days", 0, 30, 0)
    phys_hlth = col10.slider("Days of poor physical health in the past 30 days", 0, 30, 0)

    healthcare = st.checkbox("I have health insurance or healthcare coverage", value=True)
    no_doc     = st.checkbox("There was a time I needed a doctor but couldn't afford it")

    st.markdown("---")

    # --- Predict ---
    row = {
        "HighBP": int(high_bp), "HighChol": int(high_chol),
        "CholCheck": int(chol_check), "BMI": round(bmi, 1),
        "Smoker": int(smoker), "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heart), "PhysActivity": int(phys_act),
        "Fruits": int(fruits), "Veggies": int(veggies),
        "HvyAlcoholConsump": int(alcohol), "AnyHealthcare": int(healthcare),
        "NoDocbcCost": int(no_doc), "GenHlth": gen_hlth_val,
        "MentHlth": ment_hlth, "PhysHlth": phys_hlth,
        "DiffWalk": int(diff_walk), "Sex": 1 if sex == "Male" else 0,
        "Age": age_to_brfss(age), "Education": education[1],
        "Income": income[1],
    }
    X_input = pd.DataFrame([row])[feature_cols]
    proba = float(model.predict_proba(X_input)[0, 1])
    pred  = int(proba >= threshold)
    label, color, advice = risk_level(pred, proba)

    st.subheader("Your result")
    st.markdown(
        f"""
        <div style="border:2px solid {color}; border-radius:12px; padding:22px; text-align:center;">
            <div style="font-size:2rem; font-weight:700; color:{color};">{label}</div>
            <div style="font-size:1.05rem; margin-top:10px; color:#555;">{advice}</div>
            <div style="font-size:0.85rem; margin-top:14px; color:#999;">Risk score: {proba:.0%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(proba, 1.0))

    with st.expander("What's contributing to your score?"):
        factors = []
        if bmi >= 30:        factors.append(("BMI in obese range", "high"))
        elif bmi >= 25:      factors.append(("BMI in overweight range", "medium"))
        if age >= 45:        factors.append(("Age 45 or older", "medium"))
        if high_bp:          factors.append(("High blood pressure", "high"))
        if high_chol:        factors.append(("High cholesterol", "medium"))
        if heart:            factors.append(("Heart disease history", "high"))
        if stroke:           factors.append(("Stroke history", "high"))
        if smoker:           factors.append(("Smoking history", "medium"))
        if not phys_act:     factors.append(("No recent physical activity", "medium"))
        if gen_hlth_val >= 4:factors.append(("Fair or poor general health", "high"))
        if diff_walk:        factors.append(("Difficulty walking", "medium"))
        if not factors:
            st.write("No major risk factors in your answers.")
        else:
            for name, level in factors:
                st.write(f"{'🔴' if level == 'high' else '🟡'} {name}")

    st.markdown("---")
    st.caption(
        "This tool uses a logistic regression model trained on CDC BRFSS 2015 data. "
        "It is for educational purposes only and is not a medical diagnosis. "
        "Please consult a healthcare professional if you have concerns."
    )


if __name__ == "__main__":
    main()
