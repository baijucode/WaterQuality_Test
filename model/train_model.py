# model/train_model.py
# Improved version — uses XGBoost + class balancing + tuning

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── 1. Load dataset ───────────────────────────────────────────────────────────
df = pd.read_csv("data/water_potability.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Split features and target ──────────────────────────────────────────────
X = df.drop("Potability", axis=1)
y = df["Potability"]

# ── 3. Handle missing values ──────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")  # median is better than mean for skewed data
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ── 4. Fix class imbalance with SMOTE ─────────────────────────────────────────
# Problem: dataset has more Unsafe samples than Safe ones
# SMOTE creates synthetic Safe samples to balance it out
print(f"\nBefore SMOTE — Unsafe: {sum(y==0)}, Safe: {sum(y==1)}")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"After SMOTE  — Unsafe: {sum(y_balanced==0)}, Safe: {sum(y_balanced==1)}")

# ── 5. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

# ── 6. Define models ──────────────────────────────────────────────────────────
# Model A — Tuned Random Forest
rf = RandomForestClassifier(
    n_estimators=300,      # more trees = more stable
    max_depth=15,          # prevents overfitting
    min_samples_split=5,
    random_state=42
)

# Model B — XGBoost (stronger algorithm)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,    # learns slowly but more accurately
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)

# ── 7. Combine both models (Voting Classifier) ────────────────────────────────
# Like asking two experts and taking the majority vote
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    voting="soft"          # uses probability, more accurate than hard vote
)

print("\nTraining models... (this may take 1-2 minutes)")
ensemble.fit(X_train, y_train)

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
predictions = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=["Unsafe", "Safe"]))

# ── 9. Save model + imputer ───────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": ensemble, "imputer": imputer}, f)

print("\nmodel.pkl saved successfully!")