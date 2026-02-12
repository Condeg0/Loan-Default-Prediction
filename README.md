# Credit Risk Optimization Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Pipeline-orange)
![Financial Modeling](https://img.shields.io/badge/Domain-Quant%20Finance-green)

## 📌 Executive Summary
**Goal:** Transition consumer lending from "Default Prediction" (binary classification) to "Profit Maximization" (financial optimization).

**Result:** Developed a credit scoring engine that prioritizes **Net Profit** over standard accuracy metrics. By optimizing the approval threshold based on the specific P&L structure of the loan portfolio, this model identifies **$1.26M in unrealized value** compared to a naive strategy.

**Business Impact:**
* **Projected Value:** $1,264,043 incremental profit on the test set.
* **Risk Policy:** Recommends a strict approval cutoff at the **82.7th percentile** of applicants.
* **ROI Lift:** Improves portfolio profitability by **~40%** compared to the baseline "Approve All" strategy.
* **Production Model:** Logistic Regression (AUC 0.685) selected over XGBoost/RandomForest for regulatory interpretability and robust generalization.

---

## 🧮 Financial Methodology
Unlike standard ML competitions, this project grounds predictions in financial reality by reverse-engineering the asset structure and defining a custom objective function.

### 1. Reverse-Engineering the Asset
The dataset provided monthly installments but not the loan principal. I derived the Principal ($P$) using the **Present Value of an Annuity** formula, assuming the `installment` ($A$) and `int_rate` ($r$) are constant:

$$P = \frac{r \times A}{1 - (1+r)^{-N}}$$

*Implementation: `src/data_pipeline.py` (Class: `CalculateLoanAmount`)*

### 2. The Profit Function (P&L)
The model's objective is to maximize the sum of individual loan outcomes based on the confusion matrix:

* **True Negative (Good Loan):**
    $$Profit = (\text{Installment} \times \text{Term}) - \text{Principal}$$
    *Represents Total Interest Income.*

* **False Negative (Default):**
    $$Profit = (\text{Principal} \times \text{Recovery Rate}) - \text{Principal}$$
    *Represents Loss of Capital (assuming 10% recovery rate).*

* **Rejected Loan:**
    $$Profit = 0$$
    *Represents Opportunity Cost.*

---

## 🏗 System Architecture
This project utilizes a modular, production-ready architecture rather than monolithic notebooks.

* `src/data_pipeline.py`: Custom Scikit-Learn transformers for financial feature engineering with zero data leakage.
* `src/training.py`: Modular training logic using `RandomizedSearchCV` for hyperparameter tuning.
* `src/evaluation.py`: Business-centric evaluation suite calculating **Profit Curves**, **Expected Loss**, and **Gini Coefficients**.
* `notebooks/`:
    * `01_Financial_Exploration.ipynb`: EDA focused on the relationship between FICO, Interest Rates, and Default.
    * `02_model_development.ipynb`: End-to-end training and strategy optimization pipeline.

---

## 📊 Key Insights & Strategy

### 1. The "Strategy Curve" (Profit vs. Risk)
A standard model optimizes for F1-Score or Accuracy. This engine optimizes for **Net Profit**. By plotting the "Efficient Frontier" of approval rates, we determined that rejecting the bottom 17.3% of applicants (based on predicted probability) maximizes return on capital while minimizing exposure to asymmetric downside risk.

### 2. Risk Drivers
* **FICO Score:** The strongest monotonic predictor of creditworthiness.
* **Inquiries (Last 6 Months):** High velocity of credit-seeking correlates strongly with default (distress signal).
* **Interest Rate:** Confirms that riskier loans are priced higher, but the risk premium is often insufficient to offset the Default Rate in the lowest deciles.

---

## ⚠️ Assumptions & Future Improvements
To ensure a robust but feasible MVP, the current engine relies on the following assumptions. In a live production environment, these would be the immediate next steps:

1.  **Cost of Capital:** The current model assumes a 0% cost of funds. Future iterations will subtract `LIBOR + Spread` from the profit function to reflect the true cost of lending money.
2.  **Time Value of Money (TVM):** Cash flows are currently treated as nominal values. A Discounted Cash Flow (DCF) model should be applied to value long-term (60-month) loans accurately.
3.  **Prepayment Risk:** The model assumes full term adherence. A "Prepayment Hazard Model" is needed to account for borrowers who pay off early, reducing expected interest income.

---

## 🛠 Usage

**1. Installation**
```bash
git clone [https://github.com/condeg0/Credit-Risk-Optimization.git](https://github.com/condeg0/Credit-Risk-Optimization.git)
pip install -r requirements.txt

```

**2. Run the Analysis**
To generate the financial analysis and train the models:

```bash
# Generate the Financial Analysis Report
jupyter nbconvert --to html notebooks/01_Financial_Exploration.ipynb

# Execute the training and strategy pipeline
jupyter nbconvert --to html notebooks/02_model_development.ipynb

```

## 📂 Repository Structure

```text
├── data/               # Raw and processed datasets
├── notebooks/          # Analysis and Prototyping
├── src/                # Production Code
│   ├── __init__.py
│   ├── config.py       # Global Configuration (Paths, Constants)
│   ├── data_pipeline.py# Feature Engineering Transformers
│   ├── evaluation.py   # Profit Calculation & Metrics
│   └── training.py     # Model Training Wrappers
└── README.md           # Project Documentation

```
