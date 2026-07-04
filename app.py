"""
Heart Disease Risk Prediction API
Serves the trained Logistic Regression model with a tuned decision threshold
that prioritizes recall (catching true positive cases) over raw accuracy.

Run: uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart disease risk from clinical features using a "
                 "Logistic Regression model tuned for high recall.",
    version="1.0.0",
)

model = joblib.load("heart_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")
with open("model_metadata.json") as f:
    METADATA = json.load(f)

THRESHOLD = METADATA["chosen_threshold"]
USES_SCALER = METADATA["uses_scaler"]


class PatientFeatures(BaseModel):
    age: int = Field(..., example=58, description="Age in years")
    sex: int = Field(..., example=1, description="1 = male, 0 = female")
    cp: int = Field(..., example=0, description="Chest pain type (0-3)")
    trestbps: int = Field(..., example=140, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., example=289, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., example=0, description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: int = Field(..., example=1, description="Resting ECG results (0-2)")
    thalach: int = Field(..., example=140, description="Max heart rate achieved")
    exang: int = Field(..., example=0, description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., example=1.2, description="ST depression induced by exercise")
    slope: int = Field(..., example=1, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., example=0, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., example=2, description="Thalassemia type (0-3)")


class PredictionResponse(BaseModel):
    risk_probability: float
    predicted_class: int
    risk_label: str
    threshold_used: float
    model_name: str


def preprocess(patient: PatientFeatures) -> pd.DataFrame:
    raw = pd.DataFrame([patient.dict()])
    categorical_cols = ["cp", "restecg", "slope", "ca", "thal", "sex", "fbs", "exang"]
    encoded = pd.get_dummies(raw, columns=categorical_cols, drop_first=True)
    # align columns with training set (missing dummy columns -> 0)
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[feature_columns]
    return encoded


@app.get("/")
def root():
    return {
        "service": "Heart Disease Risk Prediction API",
        "model": METADATA["best_model"],
        "roc_auc": METADATA["roc_auc"],
        "chosen_threshold": THRESHOLD,
        "note": "Threshold tuned to prioritize recall over raw accuracy — "
                "missing a true case is costlier than a false alarm.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    X = preprocess(patient)
    X_input = scaler.transform(X) if USES_SCALER else X
    proba = float(model.predict_proba(X_input)[:, 1][0])
    predicted_class = int(proba >= THRESHOLD)
    return PredictionResponse(
        risk_probability=round(proba, 4),
        predicted_class=predicted_class,
        risk_label="High Risk" if predicted_class == 1 else "Low Risk",
        threshold_used=THRESHOLD,
        model_name=METADATA["best_model"],
    )
