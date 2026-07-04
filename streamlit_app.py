"""
Heart Disease Risk Prediction — Streamlit frontend
Loads the trained model directly (no separate API call needed) and lets a
clinician-style user enter patient details to get a risk prediction.

Run: streamlit run streamlit_app.py
"""
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, feature_columns, metadata

model, scaler, feature_columns, metadata = load_artifacts()
THRESHOLD = metadata["chosen_threshold"]
USES_SCALER = metadata["uses_scaler"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Risk Prediction")
st.caption(
    f"Model: {metadata['best_model']} · ROC-AUC: {metadata['roc_auc']:.3f} · "
    f"Decision threshold tuned to {THRESHOLD} for higher recall "
    "(catching more true at-risk patients, at the cost of a few extra false alarms)."
)

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("patient_form"):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=58)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest Pain Type", options=[0, 1, 2, 3],
            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=60, max_value=250, value=140)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=289)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox(
            "Resting ECG Result", options=[0, 1, 2],
            format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x],
        )

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=140)
        exang = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment", options=[0, 1, 2],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
        )
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia", options=[0, 1, 2, 3],
            format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][x],
        )

    submitted = st.form_submit_button("Predict Risk", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if submitted:
    raw = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }])

    categorical_cols = ["cp", "restecg", "slope", "ca", "thal", "sex", "fbs", "exang"]
    encoded = pd.get_dummies(raw, columns=categorical_cols, drop_first=True)
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[feature_columns]

    X_input = scaler.transform(encoded) if USES_SCALER else encoded
    proba = float(model.predict_proba(X_input)[:, 1][0])
    predicted_class = int(proba >= THRESHOLD)

    st.divider()
    st.subheader("Result")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Risk Probability", f"{proba:.1%}")
    with c2:
        st.metric("Decision Threshold", f"{THRESHOLD:.0%}")

    if predicted_class == 1:
        st.error(f"⚠️ **High Risk** — predicted probability of heart disease: {proba:.1%}")
    else:
        st.success(f"✅ **Low Risk** — predicted probability of heart disease: {proba:.1%}")

    st.progress(min(proba, 1.0))
    st.caption(
        "This tool is trained on the UCI Heart Disease dataset for portfolio/demo "
        "purposes only and is not a substitute for medical diagnosis."
    )
