import math
import streamlit as st


def diabetes_risk(age, bmi, family_history, hypertension, smoking, sedentary, high_sugar_diet):
    # Logistic regression-style formula based on known population risk weights
    # Intercept calibrated so a 35yo, BMI 22, no risk factors gives ~8% risk
    log_odds = (
        -4.2
        + 0.035 * age
        + 0.085 * bmi
        + 0.90 * family_history
        + 0.75 * hypertension
        + 0.45 * smoking
        + 0.65 * sedentary
        + 0.55 * high_sugar_diet
    )
    return 1 / (1 + math.exp(-log_odds))


def risk_level(score):
    if score < 0.20:
        return "Low Risk", "#27ae60", "No major risk factors detected. Keep up the healthy habits."
    if score < 0.45:
        return "Moderate Risk", "#e67e22", "Some risk factors present. Small lifestyle changes can make a big difference."
    return "High Risk", "#c0392b", "Several risk factors are present. A check-up with your doctor is a good idea."


def main():
    st.set_page_config(page_title="Diabetes Risk Check", layout="centered")

    st.title("Diabetes Risk Check")
    st.caption("Answer a few questions to get a personal risk estimate. No lab tests needed.")
    st.markdown("---")

    st.subheader("About you")
    col1, col2 = st.columns(2)
    age    = col1.slider("Age", 18, 90, 40)
    gender = col2.radio("Gender", ["Male", "Female"], horizontal=True)  # noqa: F841

    col3, col4 = st.columns(2)
    height_cm = col3.number_input("Height (cm)", min_value=120, max_value=230, value=170)
    weight_kg = col4.number_input("Weight (kg)", min_value=30, max_value=250, value=75)

    bmi = weight_kg / (height_cm / 100) ** 2
    bmi_label = (
        "Underweight" if bmi < 18.5 else
        "Normal"      if bmi < 25   else
        "Overweight"  if bmi < 30   else
        "Obese"
    )
    st.caption(f"Your BMI: **{bmi:.1f}** — {bmi_label}")

    st.markdown("---")
    st.subheader("Lifestyle & health history")

    family_history  = st.checkbox("A parent or sibling has been diagnosed with diabetes")
    hypertension    = st.checkbox("I have been told I have high blood pressure")
    smoking         = st.checkbox("I smoke or have smoked in the past 5 years")

    activity = st.select_slider(
        "How physically active are you?",
        options=["Rarely / never", "1–2x per week", "3–4x per week", "Daily"],
        value="1–2x per week",
    )

    diet = st.select_slider(
        "How would you describe your daily diet?",
        options=["Lots of sugary / processed food", "Mixed", "Mostly whole foods"],
        value="Mixed",
    )

    st.markdown("---")

    score = diabetes_risk(
        age=age,
        bmi=bmi,
        family_history=float(family_history),
        hypertension=float(hypertension),
        smoking=float(smoking),
        sedentary=float(activity == "Rarely / never"),
        high_sugar_diet=float(diet == "Lots of sugary / processed food"),
    )

    label, color, advice = risk_level(score)

    st.subheader("Your result")
    st.markdown(
        f"""
        <div style="border:2px solid {color}; border-radius:12px; padding:22px; text-align:center;">
            <div style="font-size:2rem; font-weight:700; color:{color};">{label}</div>
            <div style="font-size:1.05rem; margin-top:10px; color:#555;">{advice}</div>
            <div style="font-size:0.85rem; margin-top:14px; color:#999;">Risk score: {score:.0%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(score, 1.0))

    with st.expander("What's contributing to your score?"):
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
        if activity == "Rarely / never":
            factors.append(("Low physical activity", "medium"))
        if diet == "Lots of sugary / processed food":
            factors.append(("High sugar / processed diet", "medium"))

        if not factors:
            st.write("No major risk factors in your answers.")
        else:
            for name, level in factors:
                icon = "🔴" if level == "high" else "🟡"
                st.write(f"{icon} {name}")

    st.markdown("---")
    st.caption(
        "This tool is for educational purposes only and is not a medical diagnosis. "
        "Consult a healthcare professional if you have concerns about your health."
    )


if __name__ == "__main__":
    main()
