# Data Science & Machine Learning Portfolio

Welcome to my machine learning repository! This project portfolio contains end-to-end Python implementations for multi-class classification and continuous regression pipelines utilizing **Scikit-Learn**, **Pandas**, **Seaborn**, and **Matplotlib**.

Both projects are built and fully documented inside individual Jupyter Notebooks (`.ipynb`) located within this repository.

---

## 📂 Repository Contents & Navigation

### 1. 🌸 [Project 1: Multi-Class Benchmark Classification Pipeline](./)
* **Goal:** Build an end-to-end supervised classification pipeline to accurately identify species categories.
* **Dataset:** Classic Iris physical measurements dataset (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`).
* **Methodology:**
  * Exploratory multivariate pairplots via Seaborn.
  * Standardized feature normalization using `StandardScaler`.
  * Train/Test data segmentation utilizing an **80/20 split**.
  * Comparative model benchmarking between **Logistic Regression** and a **Decision Tree Classifier**.
* **Deliverables Displayed:** Native classification metric reports (Precision, Recall, F1-Score), side-by-side Confusion Matrix Heatmaps, and 2D decision boundary visualization fields.

### 2. 🏠 [Project 2: Real Estate Housing Price Prediction](./)
* **Goal:** Build a continuous regression machine learning pipeline to optimize housing price valuations.
* **Dataset:** California Housing district parameters (`median_house_value`, `median_income`, spatial coordinates, etc.).
* **Methodology:**
  * Automated data quality diagnostics scan for structural missing data loops.
  * Predictive interaction feature engineering (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`).
  * Multicollinearity mapping using custom Seaborn correlation heatmaps.
  * Categorical One-Hot Encoding transformation paths on geographic values.
  * Hyperparameter optimization tuning via a **5-Fold Cross-Validated `GridSearchCV`**.
  * Structural comparative performance testing across **OLS Linear Regression**, **Ridge (L2)**, and **Lasso (L1)** paths.
* **Deliverables Displayed:** Comparative model optimization metric tables evaluating RMSE, MAE, and $R^2$, paired alongside dual-panel Actual vs. Predicted scatter models and Residual Error Distribution overlays.

---

## 🛠️ Environment Setup & System Dependencies

This workspace runs within a localized **Anaconda** deployment context. To pull down and run these notebooks locally on your machine, initialize the following environment instructions:

1. **Clone the Portfolio Repository:**
   ```bash
   git clone https://github.com
   cd Datheon_Internship
   ```

2. **Verify Required Core Libraries:**
   Ensure your Anaconda workspace or environment terminal has the data science stack installed:
   ```bash
   conda install numpy pandas matplotlib seaborn scikit-learn -y
   ```

3. **Launch the Workspace:**
   ```bash
   jupyter notebook
   ```
   Open either individual project notebook file directly within your local Jupyter browser window and execute cells sequentially (`Shift + Enter`). All charts, plots, and performance printouts will dynamically compile.

