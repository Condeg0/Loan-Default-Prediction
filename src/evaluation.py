import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import src.config as cfg

# --- FINANCIAL LOGIC ---

def calculate_loan_financials(df):
    """
    Derives financial metrics for strategy curve.
    """
    data = df.copy()
    monthly_rate = data['int.rate'] / 12
    term = cfg.LOAN_TERM_MONTHS
    
    # Defensive: Calculate principal if missing
    if 'principal' not in data.columns:
        data['principal'] = data['installment'] * (
            1 - (1 + monthly_rate)**(-term)
        ) / monthly_rate
    
    # Profit Formulas
    data['profit.fully.paid'] = (data['installment'] * term) - data['principal']
    data['profit.default'] = -data['principal']
    
    recovery_amount = data['principal'] *cfg.RECOVERY_RATE
    data['profit.default'] = recovery_amount - data['principal']
    
    return data

def calculate_portfolio_profit(y_true, y_pred_proba, financial_df, threshold=0.5):
    decisions = y_pred_proba < threshold 
    portfolio = financial_df.copy()
    portfolio['approved'] = decisions
    
    conditions = [
        (portfolio['approved'] == True) & (y_true == 1),
        (portfolio['approved'] == True) & (y_true == 0),
    ]
    choices = [portfolio['profit.default'], portfolio['profit.fully.paid']]
    
    portfolio['realized.pnl'] = np.select(conditions, choices, default=0)
    return portfolio['realized.pnl'].sum(), portfolio['approved'].sum(), len(portfolio)

# --- VISUALIZATION ---

def plot_performance_dashboard(model, X_test, y_test, threshold=0.5):
    """
    Plots Confusion Matrix and ROC Curve side-by-side.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    # Create Subplots: 1 Row, 2 Columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], annot_kws={"size": 14})
    axes[0].set_title(f"Confusion Matrix (Threshold={threshold})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label (1=Default)', fontsize=12)
    axes[0].set_ylabel('True Label (1=Default)', fontsize=12)
    
    # --- Plot 2: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, color='#2c3e50', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC - Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print Text Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))

def plot_strategy_curve(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    fin_df = calculate_loan_financials(X_test)
    
    thresholds = np.linspace(0, 1, 100)
    profits = []
    approval_rates = []
    
    for t in thresholds:
        profit, n_approved, n_total = calculate_portfolio_profit(y_test, y_probs, fin_df, threshold=t)
        profits.append(profit)
        approval_rates.append(n_approved / n_total * 100)
        
    plt.figure(figsize=(10, 6))
    plt.plot(approval_rates, profits, lw=3, color='#2c3e50')
    
    max_idx = np.argmax(profits)
    max_profit = profits[max_idx]
    opt_rate = approval_rates[max_idx]
    
    plt.scatter(opt_rate, max_profit, color='red', s=100, label=f'Max Profit: ${max_profit:,.0f}')
    plt.axvline(opt_rate, color='red', linestyle='--', alpha=0.5)
    
    plt.title("Portfolio Profit vs. Approval Rate", fontsize=14, fontweight='bold')
    plt.xlabel("Percentage of Loans Approved (%)")
    plt.ylabel("Total Portfolio Profit ($)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print(f"--- BUSINESS INSIGHT ---")
    print(f"Optimal Strategy: Approve the safest {opt_rate:.1f}% of applicants.")
    print(f"Projected Profit: ${max_profit:,.2f}")

def explain_model_shap(model, X_train, X_test):
    """
    Generates SHAP Summary plot with CORRECT Variable Names.
    """
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    # Transform Data
    print("Transforming data for SHAP analysis...")
    X_test_transformed = preprocessor.transform(X_test)
    
    # Extract Feature Names
    feature_names = preprocessor.get_feature_names_out()
    
    # Compute SHAP
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test_transformed)
    except:
        X_train_transformed = preprocessor.transform(X_train.sample(100, random_state=42))
        explainer = shap.Explainer(classifier, X_train_transformed)
        shap_values = explainer(X_test_transformed).values

    # Plot
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1] 
    else:
        shap_vals_to_plot = shap_values
        
    shap.summary_plot(shap_vals_to_plot, X_test_transformed, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Impact on Default Risk)")
    plt.show()

def evaluate_full(model, X_test, y_test, X_train=None):
    # 1. Technical Performance (Side-by-Side Plots)
    plot_performance_dashboard(model, X_test, y_test)
    
    # 2. Business Performance (Strategy Curve)
    plot_strategy_curve(model, X_test, y_test)
    
    # 3. Model Explainability (SHAP)
    if X_train is not None:
        explain_model_shap(model, X_train, X_test)