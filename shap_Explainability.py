# Package Installation
# pip install shap scikit-learn matplotlib numpy pandas


# =========================
# SHAP explainability demo
# Model: Logistic Regression
# Dataset: sklearn Breast Cancer
# Plots: bar (global), beeswarm (global), waterfall (local), decision (local)
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap

# 1) Data
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2) Scale (good for LR); SHAP will then explain the scaled inputs
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3) Train model
clf = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
clf.fit(X_train_s, y_train)

# 4) Build SHAP explainer (LinearExplainer for linear/logistic models)
# link='logit' returns SHAP values in log-odds; they add up to the logit of the prediction.
explainer = shap.LinearExplainer(clf, X_train_s, feature_dependence="independent", link="logit")
shap_values = explainer.shap_values(X_test_s)          # shape: (n_samples, n_features)
base_value = explainer.expected_value                  # scalar (log-odds)

print(f"Base value (log-odds): {base_value:.4f}")

# 5) --- GLOBAL IMPORTANCE: Bar plot ---
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_s, feature_names=feature_names, plot_type='bar', show=False)
plt.title("Global Feature Importance (mean |SHAP|) — Logistic Regression")
plt.tight_layout()
plt.show()

# 6) --- GLOBAL DISTRIBUTION: Beeswarm ---
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_s, feature_names=feature_names, show=False)
plt.title("SHAP Summary (Beeswarm) — Logistic Regression")
plt.tight_layout()
plt.show()

# 7) --- LOCAL EXPLANATION: Waterfall plot for one prediction ---
# Choose a test instance to explain
i = 0  # change to inspect a different row
row_shap = shap_values[i, :]
row_features = X_test_s[i, :]

# SHAP needs a shap.Explanation object for waterfall in recent versions
# Construct Explanation with fields: values, base_values, data, feature_names
exp = shap._explanation.Explanation(
    values=row_shap,
    base_values=np.array([base_value]),
    data=row_features,
    feature_names=feature_names
)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(exp, max_display=15, show=False)
plt.title(f"Waterfall Plot — Test Row {i}")
plt.tight_layout()
plt.show()

# 8) --- LOCAL EXPLANATION: Decision plot (cumulative contributions) ---
plt.figure(figsize=(10, 6))
shap.decision_plot(
    base_value,
    row_shap,
    feature_names=feature_names,
    show=False
)
plt.title(f"Decision Plot — Test Row {i}")
plt.tight_layout()
plt.show()

# 9) Optional: Convert log-odds to probability for the same row
from scipy.special import expit
logit_pred = base_value + row_shap.sum()
prob_pred = expit(logit_pred)
print(f"Predicted probability (row {i}): {prob_pred:.3f}")
``
