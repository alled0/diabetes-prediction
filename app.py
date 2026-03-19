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

CSS = """
<style>
    [data-testid="stAppViewContainer"] { background: #f8f9fb; }
    [data-testid="stHeader"] { background: transparent; }
    .section-card {
        background: white;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #888;
        margin-bottom: 16px;
    }
    .result-card {
        border-radius: 16px;
        padding: 36px 28px;
        text-align: center;
        margin: 10px 0 20px 0;
    }
    .result-label {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .result-score {
        font-size: 1rem;
        margin-top: 6px;
        opacity: 0.7;
    }
    .result-advice {
        font-size: 1rem;
        margin-top: 14px;
        line-height: 1.5;
    }
    .gauge-track {
        background: #e9ecef;
        border-radius: 99px;
        height: 10px;
        margin: 18px 0 6px 0;
        overflow: hidden;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.5s ease;
    }
    .factor-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 7px 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.95rem;
    }
    .factor-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .bmi-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 99px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-left: 8px;
    }
    div[data-testid="stCheckbox"] label { font-size: 0.95rem !important; }
    .stSlider > label { font-size: 0.95rem !important; }
    footer { visibility: hidden; }
</style>
"""


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
    if bmi < 18.5:
        return "Underweight", "#3498db"
    if bmi < 25:
        return "Normal", "#27ae60"
    if bmi < 30:
        return "Overweight", "#e67e22"
    return "Obese", "#c0392b"


def risk_level(pred, proba):
    if pred == 0:
        return "Low Risk", "#27ae60", "#eafaf1", "No major risk factors detected. Keep up your healthy habits."
    if proba < 0.65:
        return "Moderate Risk", "#e67e22", "#fef9f0", "Some risk factors present. Small lifestyle changes can make a real difference."
    return "High Risk", "#c0392b", "#fdf2f2", "Several risk factors detected. Consider speaking with a healthcare professional."


def main():
    st.set_page_config(page_title="Diabetes Risk Check", page_icon="", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    model, threshold = load_model()
    if model is None:
        st.error(f"Model file not found: `{MODEL_PATH}`. Run the notebook save cell first.")
        return

    # Header
    st.markdown("""
        <div style="text-align:center; padding: 32px 0 24px 0;">
            <div style="font-size:2.2rem; font-weight:800; color:#1a1a2e; letter-spacing:-1px;">
                Diabetes Risk Check
            </div>
            <div style="color:#888; font-size:0.95rem; margin-top:8px;">
                Answer a few questions to estimate your diabetes risk
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Section 1 — About you
    st.markdown('<div class="section-card"><div class="section-title">About you</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    age = col1.slider("Age", 18, 90, 45)
    sex = col2.radio("Sex", ["Female", "Male"], horizontal=True)

    col3, col4 = st.columns(2)
    height_cm = col3.number_input("Height (cm)", 120, 230, 170)
    weight_kg = col4.number_input("Weight (kg)", 30, 250, 75)

    bmi = weight_kg / (height_cm / 100) ** 2
    bmi_cat, bmi_color = bmi_category(bmi)
    st.markdown(
        f'BMI: <strong>{bmi:.1f}</strong>'
        f'<span class="bmi-badge" style="background:{bmi_color}22; color:{bmi_color};">{bmi_cat}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    education = st.selectbox(
        "Highest level of education",
        options=[
            ("Never attended school or kindergarten", 1),
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
            ("$10,000 – $15,000", 2),
            ("$15,000 – $20,000", 3),
            ("$20,000 – $25,000", 4),
            ("$25,000 – $35,000", 5),
            ("$35,000 – $50,000", 6),
            ("$50,000 – $75,000", 7),
            ("More than $75,000", 8),
        ],
        format_func=lambda x: x[0],
        index=5,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2 — Health history
    st.markdown('<div class="section-card"><div class="section-title">Health history</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    high_bp    = col5.checkbox("High blood pressure")
    high_chol  = col6.checkbox("High cholesterol")
    chol_check = col5.checkbox("Cholesterol checked in past 5 years", value=True)
    stroke     = col6.checkbox("Ever had a stroke")
    heart      = col5.checkbox("Heart disease or heart attack")
    diff_walk  = col6.checkbox("Difficulty walking or climbing stairs")
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3 — Lifestyle
    st.markdown('<div class="section-card"><div class="section-title">Lifestyle</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    smoker   = col7.checkbox("Smoked 100+ cigarettes in lifetime")
    phys_act = col8.checkbox("Physically active in past 30 days", value=True)
    fruits   = col7.checkbox("Eat fruit 1+ times per day", value=True)
    veggies  = col8.checkbox("Eat vegetables 1+ times per day", value=True)
    alcohol  = col7.checkbox("Heavy alcohol use")
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4 — How you've been feeling
    st.markdown('<div class="section-card"><div class="section-title">How have you been feeling?</div>', unsafe_allow_html=True)
    gen_hlth = st.select_slider(
        "Overall health",
        options=["Excellent", "Very Good", "Good", "Fair", "Poor"],
        value="Good",
    )
    gen_hlth_val = ["Excellent", "Very Good", "Good", "Fair", "Poor"].index(gen_hlth) + 1

    col9, col10 = st.columns(2)
    ment_hlth  = col9.slider("Days of poor mental health (past 30 days)", 0, 30, 0)
    phys_hlth  = col10.slider("Days of poor physical health (past 30 days)", 0, 30, 0)
    healthcare = st.checkbox("Have health insurance or coverage", value=True)
    no_doc     = st.checkbox("Couldn't afford to see a doctor in the past year")
    st.markdown('</div>', unsafe_allow_html=True)

    # Predict
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
        "Age": age_to_brfss(age), "Education": education[1], "Income": income[1],
    }
    X_input = pd.DataFrame([row])[FEATURE_COLS]
    proba = float(model.predict_proba(X_input)[0, 1])
    pred  = int(proba >= threshold)
    label, color, bg, advice = risk_level(pred, proba)

    # Result card
    st.markdown(f"""
        <div class="result-card" style="background:{bg}; border: 2px solid {color}22;">
            <div class="result-label" style="color:{color};">{label}</div>
            <div class="result-score" style="color:{color};">Risk score: {proba:.0%}</div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{proba*100:.1f}%; background:{color};"></div>
            </div>
            <div class="result-advice" style="color:#444;">{advice}</div>
        </div>
    """, unsafe_allow_html=True)

    # Contributing factors
    factors = []
    if bmi >= 30:         factors.append(("BMI in obese range", color if pred else "#e67e22"))
    elif bmi >= 25:       factors.append(("BMI in overweight range", "#e67e22"))
    if age >= 45:         factors.append(("Age 45 or older", "#e67e22"))
    if high_bp:           factors.append(("High blood pressure", "#c0392b"))
    if high_chol:         factors.append(("High cholesterol", "#e67e22"))
    if heart:             factors.append(("Heart disease history", "#c0392b"))
    if stroke:            factors.append(("Stroke history", "#c0392b"))
    if smoker:            factors.append(("Smoking history", "#e67e22"))
    if not phys_act:      factors.append(("No recent physical activity", "#e67e22"))
    if gen_hlth_val >= 4: factors.append(("Fair or poor general health", "#c0392b"))
    if diff_walk:         factors.append(("Difficulty walking", "#e67e22"))

    with st.expander("What's contributing to your score?"):
        if not factors:
            st.write("No major risk factors found in your answers.")
        else:
            rows_html = "".join(
                f'<div class="factor-row">'
                f'<div class="factor-dot" style="background:{c};"></div>{name}'
                f'</div>'
                for name, c in factors
            )
            st.markdown(f'<div style="padding:4px 0">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center; color:#aaa; font-size:0.8rem; margin-top:32px; padding-bottom:16px;">
            Trained on CDC BRFSS 2015 data — 253,680 survey responses.
            For educational purposes only. Not a medical diagnosis.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
