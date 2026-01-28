# Package installation
# pip install aif360==0.6.1 scikit-learn pandas numpy


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# 1) Load dataset and select protected attribute `sex`
#    - Favorable label is ">50K" by default in AdultDataset
#    - Protected attribute: 'sex' (Male = privileged, Female = unprivileged by default)
adult = AdultDataset(
    protected_attribute_names=['sex'],
    privileged_classes=[['Male']],   # privileged group
)

# 2) Split into train/test
X = adult.features
y = adult.labels.ravel()

X_train, X_test, y_train, y_test, adult_train, adult_test = train_test_split(
    X, y, adult, test_size=0.3, random_state=42, stratify=y
)

# 3) Scale features (optional but helpful for LR)
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4) Train a simple classifier
clf = LogisticRegression(max_iter=200, solver='liblinear')
clf.fit(X_train_scaled, y_train)

# 5) Predict on test
y_pred = clf.predict(X_test_scaled)

# 6) Build prediction dataset in AIF360 format (so we can compute fairness metrics)
pred_test = adult_test.copy(deepcopy=True)
pred_test.labels = y_pred.reshape(-1, 1)

# 7) Define groups programmatically from dataset metadata (robust and less error-prone)
sex_idx = adult_test.protected_attribute_names.index('sex')
privileged_groups = [{'sex': adult_test.privileged_protected_attributes[sex_idx][0]}]
unprivileged_groups = [{'sex': adult_test.unprivileged_protected_attributes[sex_idx][0]}]

# 8) Compute fairness metrics
# Group-level label distribution differences (independent of a specific classifier)
bm = BinaryLabelDatasetMetric(
    dataset=adult_test,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

# Classifier-related metrics comparing true vs predicted labels
cm = ClassificationMetric(
    dataset=adult_test,
    classified_dataset=pred_test,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

# 9) Collect metrics
metrics = {
    "Statistical Parity Difference (SPD)": bm.statistical_parity_difference(),  # ~[-1,1], 0 ideal
    "Disparate Impact (DI)": bm.disparate_impact(),                             # ratio, 1 ideal; <0.8 is a concern (80% rule)
    "Equal Opportunity Difference (EOD)": cm.equal_opportunity_difference(),    # TPR gap, 0 ideal
    "Average Odds Difference (AOD)": cm.average_odds_difference(),              # (FPR+TPR)/2 gap, 0 ideal
    "Accuracy (overall)": cm.accuracy(),                                        # standard accuracy
}

# 10) Print results nicely
print("=== Fairness Metrics on Test Set (sex: Female vs Male) ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Optional: per-group positive prediction rates for quick sanity check
def pos_rate(bld, group):
    return BinaryLabelDatasetMetric(bld, unprivileged_groups=group, privileged_groups=group).base_rate()

unpriv_rate = pos_rate(pred_test, unprivileged_groups)
priv_rate = pos_rate(pred_test, privileged_groups)
print("\nPositive Prediction Rate:")
print(f"  Unprivileged (Female): {unpriv_rate:.4f}")
print(f"  Privileged (Male):    {priv_rate:.4f}")
``
