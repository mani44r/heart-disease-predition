"""
Heart Disease Risk Prediction
Dataset: UCI Heart Disease (Cleveland subset, 303 patients, 14 features)
Compares Logistic Regression, Random Forest, XGBoost; tunes decision threshold
to prioritize recall (minimizing false negatives is clinically critical).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, f1_score
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & inspect data
# ---------------------------------------------------------------------------
df = pd.read_csv("heart.csv")
print("Shape:", df.shape)
print("Target balance:\n", df["target"].value_counts(normalize=True))
print("Missing values:\n", df.isnull().sum().sum())

X = df.drop(columns=["target"])
y = df["target"]

categorical_cols = ["cp", "restecg", "slope", "ca", "thal", "sex", "fbs", "exang"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ---------------------------------------------------------------------------
# 2. Train/test split (stratified — small dataset, keep class balance)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Train & compare models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=RANDOM_STATE
    ),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    results[name] = {"model": model, "proba": auc and proba, "auc": auc}
    print(f"{name}: ROC-AUC = {auc:.3f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
best_proba = results[best_name]["proba"]
print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]['auc']:.3f})")

# ---------------------------------------------------------------------------
# 4. Threshold tuning — prioritize recall (catch more true positives)
#    Default 0.5 threshold minimizes overall error, but in a clinical
#    setting a false negative (missed diagnosis) is far costlier than
#    a false positive (extra screening). We scan thresholds and pick
#    the lowest one that still keeps precision reasonable.
# ---------------------------------------------------------------------------
thresholds = np.arange(0.2, 0.8, 0.05)
threshold_report = []
for t in thresholds:
    preds = (best_proba >= t).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    threshold_report.append({"threshold": round(t, 2), "recall": round(recall, 3),
                              "precision": round(precision, 3), "false_negatives": int(fn)})

print("\nThreshold sweep:")
for r in threshold_report:
    print(r)

# Chosen threshold: lowest false negatives while precision stays >= 0.75
chosen = next((r for r in threshold_report if r["precision"] >= 0.75), threshold_report[len(threshold_report)//2])
CHOSEN_THRESHOLD = chosen["threshold"]
print(f"\nChosen threshold: {CHOSEN_THRESHOLD} -> recall={chosen['recall']}, precision={chosen['precision']}")

final_preds = (best_proba >= CHOSEN_THRESHOLD).astype(int)
print("\nClassification report at chosen threshold:")
print(classification_report(y_test, final_preds))

# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, best_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"{best_name} (AUC={results[best_name]['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Heart Disease Risk Model")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)

cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max()/2 else "black")
plt.xticks([0, 1], ["No Disease", "Disease"])
plt.yticks([0, 1], ["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix @ threshold={CHOSEN_THRESHOLD}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# ---------------------------------------------------------------------------
# 6. Persist model artifacts for the FastAPI service
# ---------------------------------------------------------------------------
joblib.dump(best_model, "heart_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(list(X_encoded.columns), "feature_columns.joblib")

with open("model_metadata.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "roc_auc": results[best_name]["auc"],
        "chosen_threshold": CHOSEN_THRESHOLD,
        "recall_at_threshold": chosen["recall"],
        "precision_at_threshold": chosen["precision"],
        "model_comparison": {k: round(v["auc"], 3) for k, v in results.items()},
        "uses_scaler": best_name == "Logistic Regression",
    }, f, indent=2)

print("\nSaved: heart_model.joblib, scaler.joblib, feature_columns.joblib, model_metadata.json")
print("Saved plots: roc_curve.png, confusion_matrix.png")
