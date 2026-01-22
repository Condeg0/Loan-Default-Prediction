# Loan-Default-Prediction

![Scikit-Learn](https://img.shields.io/badge/Sklearn-Pipeline-orange)

## 📌 Executive Summary
**Goal:** Optimize a consumer lending portfolio by replacing naive "Accuracy" metrics with a profit-maximization strategy.

**Result:** Developed a credit scoring engine that optimizes for **Net Profit** rather than raw accuracy. While complex models (XGBoost & RandomForest) were tested, the final production model utilizes **Logistic Regression (AUC 0.685)** for its superior interpretability and test-set performance.

**Business Impact:**
* **Projected Value:** Identified **$1,264,043** in potential value.
* **Risk Policy:** Recommends a strict approval cutoff at the top **82.7%** of applicants.
* **ROI Lift:** This strategy improves portfolio profitability by **~40%** compared to the naive "Approve All" baseline.

## 🏗 System Architecture
This project moves beyond standard notebooks into a **production-ready modular architecture**:

* `src/data_pipeline.py`: Custom Scikit-Learn transformers for financial feature engineering (e.g., deriving Principal from Annuity formulas) with zero data leakage.
* `src/training.py`: Modular training logic using `RandomizedSearchCV` for hyperparameter tuning.
* `src/evaluation.py`: Business-centric evaluation suite calculating **Profit Curves** and **Expected Loss**.

## 📊 Key Insights & Performance

### 1. The "Strategy Curve" (Profit vs. Risk)
A standard model optimizes for F1-Score. This engine optimizes for **Net Profit**. By plotting the "Efficient Frontier" of approval rates, we determined that rejecting the riskiest segment of applicants maximizes return on capital while minimizing exposure to "Likely Default" profiles.

### 2. Model Performance
* **ROC-AUC Score:** **0.685** (Demonstrating reliable separation between 'Fully Paid' and 'Default' classes).
* **Primary Risk Drivers:**
    1.  **FICO Score:** The strongest predictor of creditworthiness.
    2.  **Inquiries (Last 6 Months):** High velocity of credit-seeking correlates strongly with default.
    3.  **Interest Rate:** Confirms that riskier loans are priced higher, but often not high enough to offset the default risk.

## 🛠 Usage

**1. Installation**
```bash
git clone [https://github.com/condeg0/Credit-Risk-Optimization.git](https://github.com/condeg0/Credit-Risk-Optimization.git)
pip install -r requirements.txt

# Generate the Financial Analysis Report
jupyter nbconvert --to html notebooks/01_Financial_Exploration.ipynb

## Execute the training and strategy pipeline
jupyter nbconvert --to html notebooks/02_model_development.ipynb
```


# 📂 Repository Structure

├── data/               # Raw and processed financial data
├── notebooks/          # Analysis and experimentation (EDA, Modeling)
├── src/                # Production source code
│   ├── __init__.py     # Package initialization
│   ├── config.py       # Centralized configuration (Assumptions, Constants)
│   ├── data_pipeline.py# Feature Engineering & Preprocessing Pipelines
│   ├── evaluation.py   # Profit & Logic Calculations
│   └── training.py     # Model Training & Tuning Logic
└── README.md           # Project Documentation
