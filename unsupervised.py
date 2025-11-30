# ======================================================
#   Isolation Forest Anomaly Detection
#   Using unsupervised cleaned data
# ======================================================

import pandas as pd
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------
# 1. Load dataset (no labels)
# ------------------------------------------------------

print("📥 Loading unsupervised data...")
X = pd.read_csv("X_unsupervised.csv")

print("✔ X shape:", X.shape)


# ------------------------------------------------------
# 2. Train Isolation Forest
# ------------------------------------------------------

print("🌲 Training Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,   # 2% anomalies
    random_state=42
)

model.fit(X)

print("✔ Training complete!")


# ------------------------------------------------------
# 3. Compute anomaly score
# ------------------------------------------------------

# Score: the LOWER the score, the MORE abnormal
scores = model.score_samples(X)

# Binary anomaly flag
# -1 = anomaly
#  1 = normal
labels = model.predict(X)

print("🔍 Example anomaly scores:", scores[:10])


# ------------------------------------------------------
# 4. Save results
# ------------------------------------------------------

# df_out = pd.DataFrame({
#     "anomaly_score": scores,
#     "is_anomaly": labels  # -1 = anomaly, 1 = normal
# })

df_out = pd.DataFrame({
    "txId": X.index,             # ★ 加上 txId
    "anomaly_score": scores,
    "is_anomaly": labels
})

df_out.to_csv("if_anomaly_scores.csv", index=False)

print("💾 Saved: if_anomaly_scores.csv")
