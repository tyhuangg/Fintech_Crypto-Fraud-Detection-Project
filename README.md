# 🧊 Crypto Fraud Detection — Logistic Regression + Isolation Forest  
利用 Elliptic++ Dataset 完成的完整「比特幣可疑交易偵測」專案

這是一個以 **Elliptic++ (2022)** 資料集為基礎的數據分析與機器學習實作。  
本專案目標是：

- 偵測「與已知詐騙相似的交易」  
- 偵測「未標記但具有異常行為的可疑交易」  
- 透過模型融合（Risk Fusion）給出 **最終風險分數排行榜**

---

## 📌 專案亮點 Summary

✔ **資料清理（Data Cleaning）**  
✔ **特徵工程（Feature Engineering）**  
✔ **監督式模型：Logistic Regression（fraud_score）**  
✔ **非監督式模型：Isolation Forest（anomaly_score）**  
✔ **風險融合模型：Risk Fusion（fraud_risk_score）**  
✔ **可疑交易排名（Top Risky Transactions Ranking）**  
✔ **完整可重現的 pipeline（.py 程式 + 輸出 CSV）**

---

## 📂 專案結構（Repository Structure）
```
finalProj/
├─ cleaned_merged_dataset.csv # 清理後的完整特徵 + 標籤
├─ dataMerge.py # Merge tx_features + tx_classes 的程式
├─ supervised.py # Logistic Regression（監督式模型）
├─ unsupervised.py # Isolation Forest（非監督式模型）
├─ finalRiskScore.py # Risk Fusion 模型（LR + IF）
├─ lr_predictions.csv # LR 輸出的 fraud_score
├─ if_anomaly_scores.csv # IF 輸出的 anomaly_score
├─ risk_fusion_scores.csv # 最終風險分數 + 排名
├─ X_supervised.csv
├─ y_supervised.csv
├─ X_unsupervised.csv
├─ txs_classes.csv # 原始標籤（licit / illicit / unknown）
├─ txs_features.csv # 原始特徵（地區特徵 + 彙總特徵 + BTC 特徵）
└─ txs_edgelist.csv # 交易網路邊資料（未使用模型）
```

---

## 📊 模型介紹（Models Overview）

### 1️⃣ Logistic Regression（監督式）
- 使用 Elliptic++ 提供的交易標籤：  
  - **1 = illicit（詐騙）**  
  - **0 = licit（合法）**  
- 產出 `fraud_score`（0〜1）

### 2️⃣ Isolation Forest（非監督式）
- 不使用任何標籤  
- 偵測異常行為  
- 產出：
  - `anomaly_score`
  - `is_anomaly`（1 = 可疑）

### 3️⃣ Risk Fusion（最終模型）
融合：

- `fraud_score`（LR）
- `anomaly_score_scaled`（IF）

最終輸出：

- `fraud_risk_score`
- `risk_rank`
- 完整排名（1 = 最可疑）

---

## 🔧 如何執行（Running the Project）

### 🔹 1. 資料清理合併
```bash
python3 dataMerge.py

**### 🔹 2. 監督式模型（Logistic Regression）**
```bash
python3 supervised.py

**### 🔹 3. 非監督式模型（Isolation Forest）**
```bash
python3 unsupervised.py

**### 🔹 4. 風險融合模型（最終分數）**
```bash
python3 finalRiskScore.py

**## 最終產出（Final Outputs）**
| 檔案                         | 內容                               |
| -------------------------- | -------------------------------- |
| **lr_predictions.csv**     | Logistic Regression fraud scores |
| **if_anomaly_scores.csv**  | Isolation Forest anomaly scores  |
| **risk_fusion_scores.csv** | Final risk score + ranking       |

**## 資料來源（Dataset）**
本專案使用：

Elliptic++ Transactions Dataset — SIGKDD 2022

原始資料來源：
https://github.com/git-disl/EllipticPlusPlus
