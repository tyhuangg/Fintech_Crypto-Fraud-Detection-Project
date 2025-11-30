# ========================================================
#   Elliptic++ Transactions Dataset - Data Cleaning Notebook
#   Author: Claudia (Based on ChatGPT Preprocessing Pipeline)
# ========================================================

import pandas as pd
import numpy as np

# --------------------------------------------------------
# 1. Read CSV Files
# --------------------------------------------------------

# ⚠️ 記得把路徑改成你自己的資料夾位置
features_path = "txs_features.csv"
classes_path = "txs_classes.csv"

print("📥 Loading CSV files...")
features = pd.read_csv(features_path)
classes = pd.read_csv(classes_path)

print("✔ features shape:", features.shape)
print("✔ classes shape:", classes.shape)


# --------------------------------------------------------
# 2. Merge Feature + Label
# --------------------------------------------------------

print("🔄 Merging on txId...")

df = features.merge(classes, on="txId", how="left")

print("✔ merged shape:", df.shape)
print(df.head())


# --------------------------------------------------------
# 3. Remove unknown labels (class = 3)
# --------------------------------------------------------

print("🧹 Removing unknown (class = 3)...")

df_clean = df[df["class"] != 3].copy()

print("✔ after removing unknown:", df_clean.shape)
print(df_clean["class"].value_counts())


# --------------------------------------------------------
# 4. Convert labels: licit → 0, illicit → 1
# --------------------------------------------------------

print("🎯 Mapping class → binary label...")

df_clean["label"] = df_clean["class"].map({
    1: 0,   # licit
    2: 1    # illicit
})

print(df_clean[["class", "label"]].head())


# --------------------------------------------------------
# 5. Prepare X (features) and y (labels) for Logistic Regression
# --------------------------------------------------------

print("📊 Preparing X, y for Logistic Regression...")

# Drop non-feature columns
non_feature_cols = ["txId", "class", "label"]

X = df_clean.drop(columns=non_feature_cols)
y = df_clean["label"]

print("✔ X shape:", X.shape)
print("✔ y shape:", y.shape)


# --------------------------------------------------------
# 6. Prepare data for Isolation Forest (UNSUPERVISED)
# --------------------------------------------------------

print("🌲 Preparing data for Isolation Forest...")

# For IF, we do NOT remove unknowns
X_if = df.drop(columns=["txId", "class"])  # only numerical features
print("✔ X_if shape:", X_if.shape)


# --------------------------------------------------------
# 7. Save outputs to CSV (給模型組)
# --------------------------------------------------------

print("💾 Saving cleaned datasets...")

df_clean.to_csv("cleaned_merged_dataset.csv", index=False)
X.to_csv("X_supervised.csv", index=False)
y.to_csv("y_supervised.csv", index=False)
X_if.to_csv("X_unsupervised.csv", index=False)

print("🎉 Done! Files generated:")
print(" - cleaned_merged_dataset.csv")
print(" - X_supervised.csv")
print(" - y_supervised.csv")
print(" - X_unsupervised.csv")
