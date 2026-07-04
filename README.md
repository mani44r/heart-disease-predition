# Heart Disease Risk Prediction

Predicts the presence of heart disease from clinical features using the UCI
Heart Disease (Cleveland) dataset. Compares three classifiers and tunes the
decision threshold to prioritize **recall** over raw accuracy, since a missed
diagnosis (false negative) is clinically far costlier than a false alarm.

## Dataset
- UCI Heart Disease dataset, Cleveland subset — 303 patients, 13 clinical
  features (age, sex, chest pain type, resting BP, cholesterol, ECG results,
  max heart rate, exercise-induced angina, ST depression, etc.)
- Binary target: presence (1) / absence (0) of heart disease
- No missing values in this subset; class balance ~55/45

## Approach
1. **EDA** — checked class balance and feature distributions
2. **Preprocessing** — one-hot encoded categorical features (chest pain type,
   ECG results, slope, thal, etc.), standardized numeric features
3. **Modeling** — trained and compared Logistic Regression, Random Forest,
   and XGBoost on ROC-AUC
4. **Threshold tuning** — swept decision thresholds from 0.2–0.8 and selected
   the lowest threshold that still keeps precision ≥ 0.75, maximizing recall
   (minimizing missed true-positive cases)
5. **Deployment** — trained model served via a FastAPI `/predict` endpoint

## Results

| Model | ROC-AUC |
|---|---|
| **Logistic Regression** | **0.913** |
| Random Forest | 0.880 |
| XGBoost | 0.876 |

At the default 0.5 threshold, the best model misses 4 of 33 true positive
cases in the test set. Lowering the threshold to **0.30** raises recall to
**93.9%** (only 2 missed cases) while precision stays at a reasonable 75.6% —
a deliberate tradeoff favoring catching more at-risk patients over avoiding
false alarms.

See `roc_curve.png` and `confusion_matrix.png` for visuals, and
`model_metadata.json` for the full metric breakdown.

## Project structure
```
heart-disease-prediction/
├── heart.csv                  # dataset
├── train_heart_model.py       # EDA, training, threshold tuning, plots
├── app.py                     # FastAPI serving the trained model
├── streamlit_app.py           # Streamlit frontend (form-based risk predictor)
├── heart_model.joblib         # trained model
├── scaler.joblib              # fitted StandardScaler (used by Logistic Regression)
├── feature_columns.joblib     # encoded feature column order
├── model_metadata.json        # metrics + chosen threshold
├── roc_curve.png
├── confusion_matrix.png
└── requirements.txt
```

## Running locally
```bash
pip install -r requirements.txt
python train_heart_model.py      # retrain from scratch (optional, artifacts already included)
uvicorn app:app --reload         # start the API at http://127.0.0.1:8000
streamlit run streamlit_app.py   # start the frontend at http://localhost:8501
```

## Frontend
`streamlit_app.py` gives a form-based UI where you enter patient details
(age, chest pain type, cholesterol, ECG results, etc.) and get back a risk
probability, a High/Low Risk label, and the decision threshold used — loads
the trained model directly, so it runs standalone without the API.

### Example request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 58, "sex": 1, "cp": 0, "trestbps": 140, "chol": 289,
    "fbs": 0, "restecg": 1, "thalach": 140, "exang": 0,
    "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2
  }'
```

### Example response
```json
{
  "risk_probability": 0.5742,
  "predicted_class": 1,
  "risk_label": "High Risk",
  "threshold_used": 0.3,
  "model_name": "Logistic Regression"
}
```

## Key takeaway
Accuracy alone is a misleading metric in healthcare classification — this
project demonstrates reasoning through the precision/recall tradeoff and
explicitly choosing an operating threshold aligned with the real-world cost
of a missed diagnosis, rather than defaulting to 0.5.
