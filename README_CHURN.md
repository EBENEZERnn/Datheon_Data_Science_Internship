# Project 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)

An intermediate-level ensemble machine learning classification pipeline built with Scikit-Learn, XGBoost, Pandas, Seaborn, and Matplotlib. This project resolves severe class imbalance issues to optimize **ROC-AUC** scores for predicting customer attrition (churn) within telecom subscription records.

## 🎯 Project Objective
Develop and benchmark advanced ensemble classifiers capable of accurately identifying at-risk subscription profiles. The pipeline explicitly converts mixed-type numeric structures, handles minority class imbalance gaps utilizing automated model sample weighting scales, and delivers actionable categorical feature importance mappings.

## 📊 Dataset Structure
The project utilizes the classic `churn.csv` dataset, which contains demographic, billing, and product subscription metrics.
* **Continuous Features:** `MonthlyCharges`, `TotalCharges`
* **Categorical Multi-Class Features:** `Contract`, `PaymentMethod`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, etc.
* **Target Classification Label:** `Churn` (Mapped: 1 = Customer Left, 0 = Customer Retained)

## 🛠️ Pipeline Architecture & Methodology

The project workspace is structured sequentially inside your Jupyter Notebook:

1. **Robust Data Type Correction:** Explicitly forces string-typed attributes (`TotalCharges`) containing hidden white spaces into true float decimals, identifying and removing newly unmasked missing data records.
2. **Pipelines & Column Transformations:** 
   * Automates structural scaling across continuous spaces using `StandardScaler`.
   * Automatically strips multicollinearity traps across textual fields utilizing `OneHotEncoder(drop='first')`.
   * Splits features into an **80/20 train/test split** with class stratification enabled.
3. **Class Imbalance Optimization:** Automatically extracts negative-to-positive feature balancing ratio targets (`scale_pos_weight`) to calibrate estimators directly during gradient boosting calculation rounds.
4. **Ensemble Model Training:** Runs a weighted bootstrap architecture (**Random Forest Classifier**) against a regularized parallel tree booster (**XGBoost Classifier**).
5. **Advanced Evaluation Metrics Suite:** Plots side-by-side Confusion Matrix Heatmaps, tracks feature weight impacts, and maps overlapping **ROC-AUC Curves** to target the assignment's $>80\%$ classification benchmark goal.

## 📋 Deliverables Included
* **Data Diagnostics Log** detailing explicit data cleaning and target imbalance ratios.
* **Overlapping ROC-AUC Curves Chart** demonstrating model optimization success exceeding the $80\%$ target benchmark.
* **Top 10 Most Influential Feature Importances Bar Chart** identifying exactly what structural customer flags trigger attrition events.
* **Dual Confusion Matrix Heatmaps** showing true vs. predicted outcomes across both ensemble architectures.
