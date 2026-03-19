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


def bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "#3b82f6"
    if bmi < 25:   return "Normal",      "#16a34a"
    if bmi < 30:   return "Overweight",  "#ea580c"
    return "Obese", "#dc2626"


def risk_config(pred, proba):
    if pred == 0:
        return "Low Risk", "#16a34a", "#f0fdf4", "No major risk factors detected. Keep up your healthy habits."
    if proba < 0.65:
        return "Moderate Risk", "#ea580c", "#fff7ed", "Some risk factors present. Small changes can make a big difference."
    return "High Risk", "#dc2626", "#fef2f2", "Several risk factors detected. Consider speaking with a healthcare professional."


def main():
    st.set_page_config(page_title="Diabetes Risk Check", layout="centered")

    st.markdown("""
        <style>
            .block-container { padding-top: 2.5rem; max-width: 720px; }
            footer { visibility: hidden; }
            .result-box {
                border-radius: 12px;
                padding: 28px 24px;
                text-align: center;
                margin: 4px 0 20px;
            }
            .risk-label { font-size: 2rem; font-weight: 800; }
            .risk-score { font-size: 0.9rem; margin-top: 4px; }
            .risk-advice { font-size: 0.95rem; margin-top: 12px; line-height: 1.6; }
            .gauge { background: #e2e8f0; border-radius: 99px; height: 8px; margin: 14px 0 2px; overflow: hidden; }
            .gauge-bar { height: 100%; border-radius: 99px; }
        </style>
    """, unsafe_allow_html=True)

    model, threshold = load_model()
    if model is None:
        st.error(f"Model file not found: `{MODEL_PATH}`.")
        return

    st.markdown("## Diabetes Risk Check")
    st.markdown("Answer a few questions to get a personalised diabetes risk estimate.")
    st.divider()

    # About you
    st.markdown("### About you")
    col1, col2 = st.columns(2)
    age = col1.slider("Age", 18, 90, 45)
    sex = col2.radio("Sex", ["Female", "Male"], horizontal=True)

    col3, col4 = st.columns(2)
    height_cm = col3.number_input("Height (cm)", 120, 230, 170)
    weight_kg = col4.number_input("Weight (kg)", 30, 250, 75)

    bmi = weight_kg / (height_cm / 100) ** 2
    bmi_cat, bmi_color = bmi_category(bmi)
    st.markdown(
        f"**BMI: {bmi:.1f}** &nbsp; "
        f"<span style='background:{bmi_color};color:white;padding:2px 10px;"
        f"border-radius:99px;font-size:0.78rem;font-weight:600;'>{bmi_cat}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    education = st.selectbox(
        "Highest level of education",
        [("Never attended school", 1), ("Elementary school", 2), ("Some high school", 3),
         ("High school graduate", 4), ("Some college or technical school", 5), ("College graduate", 6)],
        format_func=lambda x: x[0], index=3,
    )
    income = st.selectbox(
        "Annual household income",
        [("Less than $10,000", 1), ("$10,000 – $15,000", 2), ("$15,000 – $20,000", 3),
         ("$20,000 – $25,000", 4), ("$25,000 – $35,000", 5), ("$35,000 – $50,000", 6),
         ("$50,000 – $75,000", 7), ("More than $75,000", 8)],
        format_func=lambda x: x[0], index=5,
    )
    st.divider()

    # Health history
    st.markdown("### Health history")
    col5, col6 = st.columns(2)
    high_bp    = col5.checkbox("High blood pressure")
    high_chol  = col6.checkbox("High cholesterol")
    chol_check = col5.checkbox("Cholesterol checked in past 5 years", value=True)
    stroke     = col6.checkbox("Ever had a stroke")
    heart      = col5.checkbox("Heart disease or heart attack")
    diff_walk  = col6.checkbox("Difficulty walking or climbing stairs")
    st.divider()

    # Lifestyle
    st.markdown("### Lifestyle")
    col7, col8 = st.columns(2)
    smoker   = col7.checkbox("Smoked 100+ cigarettes in lifetime")
    phys_act = col8.checkbox("Physically active in past 30 days", value=True)
    fruits   = col7.checkbox("Eat fruit 1+ times per day", value=True)
    veggies  = col8.checkbox("Eat vegetables 1+ times per day", value=True)
    alcohol  = col7.checkbox("Heavy alcohol use")
    st.divider()

    # General health
    st.markdown("### How have you been feeling?")
    gen_hlth = st.select_slider(
        "Overall health",
        options=["Excellent", "Very Good", "Good", "Fair", "Poor"],
        value="Good",
    )
    gen_hlth_val = ["Excellent", "Very Good", "Good", "Fair", "Poor"].index(gen_hlth) + 1

    col9, col10 = st.columns(2)
    ment_hlth  = col9.slider("Poor mental health days (past 30)", 0, 30, 0)
    phys_hlth  = col10.slider("Poor physical health days (past 30)", 0, 30, 0)
    healthcare = st.checkbox("Have health insurance or coverage", value=True)
    no_doc     = st.checkbox("Couldn't afford to see a doctor in the past year")
    st.divider()

    # Predict
    row = {
        "HighBP": int(high_bp), "HighChol": int(high_chol), "CholCheck": int(chol_check),
        "BMI": round(bmi, 1), "Smoker": int(smoker), "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heart), "PhysActivity": int(phys_act),
        "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(alcohol),
        "AnyHealthcare": int(healthcare), "NoDocbcCost": int(no_doc), "GenHlth": gen_hlth_val,
        "MentHlth": ment_hlth, "PhysHlth": phys_hlth, "DiffWalk": int(diff_walk),
        "Sex": 1 if sex == "Male" else 0, "Age": age_to_brfss(age),
        "Education": education[1], "Income": income[1],
    }
    proba = float(model.predict_proba(pd.DataFrame([row])[FEATURE_COLS])[0, 1])
    pred  = int(proba >= threshold)
    label, color, bg, advice = risk_config(pred, proba)

    st.markdown("### Your result")
    st.markdown(f"""
        <div class="result-box" style="background:{bg};border:1.5px solid {color}44;">
            <div class="risk-label" style="color:{color};">{label}</div>
            <div class="risk-score" style="color:{color};">Risk score: {proba:.0%}</div>
            <div class="gauge">
                <div class="gauge-bar" style="width:{proba*100:.1f}%;background:{color};"></div>
            </div>
            <div class="risk-advice" style="color:#374151;">{advice}</div>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("What factors are contributing to your score?"):
        factors = []
        if bmi >= 30:         factors.append(("BMI in obese range", "#dc2626"))
        elif bmi >= 25:       factors.append(("BMI in overweight range", "#ea580c"))
        if age >= 45:         factors.append(("Age 45 or older", "#ea580c"))
        if high_bp:           factors.append(("High blood pressure", "#dc2626"))
        if high_chol:         factors.append(("High cholesterol", "#ea580c"))
        if heart:             factors.append(("Heart disease history", "#dc2626"))
        if stroke:            factors.append(("Stroke history", "#dc2626"))
        if smoker:            factors.append(("Smoking history", "#ea580c"))
        if not phys_act:      factors.append(("No recent physical activity", "#ea580c"))
        if gen_hlth_val >= 4: factors.append(("Fair or poor general health", "#dc2626"))
        if diff_walk:         factors.append(("Difficulty walking", "#ea580c"))

        if not factors:
            st.success("No major risk factors found in your answers.")
        else:
            for name, c in factors:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;padding:8px 0;"
                    f"border-bottom:1px solid #f1f5f9;color:#1e293b;'>"
                    f"<span style='width:8px;height:8px;border-radius:50%;background:{c};"
                    f"display:inline-block;flex-shrink:0;'></span>{name}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        "<p style='text-align:center;color:#94a3b8;font-size:0.78rem;margin-top:24px;'>"
        "Trained on CDC BRFSS 2015 — 253,680 survey responses. "
        "For educational purposes only. Not a medical diagnosis.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
