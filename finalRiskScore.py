# # ======================================================
# #   Notebook C: Risk Fusion Model
# #   Combine LR fraud_score + IF anomaly_score → Final Risk Score
# # ======================================================

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # ------------------------------------------------------
# # 1. Load model outputs (from Notebook A & B)
# # ------------------------------------------------------

# print("📥 Loading model outputs...")

# df_lr = pd.read_csv("lr_predictions.csv")           # fraud_score, true_label
# df_if = pd.read_csv("if_anomaly_scores.csv")       # anomaly_score, is_anomaly

# print("✔ LR shape:", df_lr.shape)
# print("✔ IF shape:", df_if.shape)


# # ------------------------------------------------------
# # 2. Align indexes if needed
# # ------------------------------------------------------

# # Note: We assume the transactions appear in the same order
# # after preprocessing. If needed, you can add txId and merge.

# # df = pd.concat([df_lr, df_if], axis=1)
# df = df_lr.merge(df_if, on="txId", how="inner")

# print("🔄 Combined shape:", df.shape)
# print(df.head())


# # ------------------------------------------------------
# # 3. Normalize Isolation Forest anomaly_score
# # ------------------------------------------------------
# # anomaly_score: LOWER = more suspicious
# # → We invert & scale it so that HIGHER = more suspicious

# print("📊 Normalizing anomaly score...")

# scaler = MinMaxScaler()

# df["anomaly_score_scaled"] = scaler.fit_transform(
#     (-df["anomaly_score"]).values.reshape(-1,1)
# )

# print(df[["anomaly_score", "anomaly_score_scaled"]].head())


# # ------------------------------------------------------
# # 4. Fusion formula — Weighted Risk Score
# # ------------------------------------------------------
# # weights can be adjusted (ex: 0.6 for LR, 0.4 for IF)

# LR_WEIGHT = 0.6
# IF_WEIGHT = 0.4

# df["fraud_risk_score"] = (
#     LR_WEIGHT * df["fraud_score"] +
#     IF_WEIGHT * df["anomaly_score_scaled"]
# )

# print("⭐ Example final fused risk score:")
# print(df["fraud_risk_score"].head())


# # ------------------------------------------------------
# # 5. Ranking transactions by risk level
# # ------------------------------------------------------

# df["risk_rank"] = df["fraud_risk_score"].rank(
#     method="first", ascending=False
# )

# df_sorted = df.sort_values("fraud_risk_score", ascending=False)

# print("📈 Top 10 highest-risk transactions:")
# print(df_sorted.head(10))


# # ------------------------------------------------------
# # 6. Save final fusion results
# # ------------------------------------------------------

# df_sorted.to_csv("risk_fusion_scores.csv", index=False)

# print("💾 Saved: risk_fusion_scores.csv")
# print("🎉 Risk fusion complete!")

# ======================================================
#   Risk Fusion Model (Final Version)
#   Combine Logistic Regression + Isolation Forest
#   with correct indexing (merge by txId)
# ======================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("📥 Loading model outputs...")

# -------------------------------------
# 1. Load LR & IF results
# -------------------------------------
df_lr = pd.read_csv("lr_predictions.csv")        # columns: txId, fraud_score, true_label
df_if = pd.read_csv("if_anomaly_scores.csv")     # columns: txId, anomaly_score, is_anomaly

print("✔ LR rows:", len(df_lr))
print("✔ IF rows:", len(df_if))

# -------------------------------------
# 2. Merge using txId (critical fix!)
# -------------------------------------
print("🔄 Merging on txId...")

df = df_lr.merge(df_if, on="txId", how="inner")

print("✔ Merged rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head())


# -------------------------------------
# 3. Normalize anomaly_score
# -------------------------------------
print("📊 Normalizing anomaly scores...")

scaler = MinMaxScaler()

# Isolation Forest: lower = more suspicious
# → invert + scale → high = suspicious
df["anomaly_score_scaled"] = scaler.fit_transform(
    (-df["anomaly_score"]).values.reshape(-1, 1)
)

print(df[["anomaly_score", "anomaly_score_scaled"]].head())


# -------------------------------------
# 4. Weighted fusion score
# -------------------------------------
LR_WEIGHT = 0.6
IF_WEIGHT = 0.4

df["fraud_risk_score"] = (
    LR_WEIGHT * df["fraud_score"] +
    IF_WEIGHT * df["anomaly_score_scaled"]
)

print("⭐ Fraud risk example:")
print(df["fraud_risk_score"].head())


# -------------------------------------
# 5. Ranking
# -------------------------------------
df["risk_rank"] = df["fraud_risk_score"].rank(
    method="first", ascending=False
)

df_sorted = df.sort_values("fraud_risk_score", ascending=False)

print("📈 Top 10 risky transactions:")
print(df_sorted.head(10))


# -------------------------------------
# 6. Save output
# -------------------------------------
df_sorted.to_csv("risk_fusion_scores.csv", index=False)

print("💾 Saved: risk_fusion_scores.csv")
print("🎉 Fusion complete — ranking is now correct!")
