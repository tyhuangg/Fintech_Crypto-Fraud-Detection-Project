import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

print("📥 Loading cleaned data...")

df = pd.read_csv("cleaned_merged_dataset.csv")

# -------------------------
# 1. Basic inspection
# -------------------------
print("🔍 Checking NaN count...")
nan_counts = df.isna().sum()
print(nan_counts[nan_counts > 0])

# -------------------------
# 2. Separate features & labels
# -------------------------
y = df["label"]
X = df.drop(columns=["label"])

print("✔ X shape:", X.shape)
print("✔ y shape:", y.shape)

# -------------------------
# 3. Fix missing values
# -------------------------
print("🛠 Handling missing values...")

# Option A：全部補 0（推薦）
X = X.fillna(0)

# （如果你想要用 median：X = X.fillna(X.median())）

print("✔ Remaining NaN after fill:", X.isna().sum().sum())

# -------------------------
# 4. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("📊 Train/Test sizes:", X_train.shape, X_test.shape)

# -------------------------
# 5. Train Logistic Regression
# -------------------------
print("🚀 Training Logistic Regression...")

model = LogisticRegression(max_iter=2000, n_jobs=-1)
model.fit(X_train, y_train)

print("✅ Training complete!")

# -------------------------
# 6. Predict
# -------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("📈 Evaluation:")
print(classification_report(y_test, y_pred))

# -------------------------
# 7. Save fraud scores
# -------------------------
print("💾 Saving LR fraud scores...")

# df_lr = pd.DataFrame({
#     "fraud_score": model.predict_proba(X)[:, 1],
#     "true_label": y
# })
df_lr = pd.DataFrame({
    "txId": df.index,             # ★ 加上 txId
    "fraud_score": model.predict_proba(X)[:, 1],
    "true_label": y
})


df_lr.to_csv("lr_predictions.csv", index=False)

print("🎉 lr_predictions.csv saved!")
