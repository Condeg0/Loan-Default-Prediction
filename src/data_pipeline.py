import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import src.config as cfg

def engineer_financial_features(df):
    """
    Applies row-wise business logic. 
    Safe to run before splitting because one row does not affect another (No Leakage).
    """
    df = df.copy()
    
    # 1. Derive Principal (Present Value Formula)
    monthly_rate = df['int.rate'] / 12
    term = cfg.LOAN_TERM_MONTHS
    
    df['principal'] = df['installment'] * (
        1 - (1 + monthly_rate)**(-term)
    ) / monthly_rate
    
    # 2. Log Income (already exists as 'log.annual.inc', but ensuring naming consistency)
    # df['log_annual_inc'] = df['log.annual.inc'] 
    
    return df

def get_preprocessing_pipeline(X_train):
    """
    Returns a ColumnTransformer that handles Numeric and Categorical data.
    Now correctly sees ALL columns including engineered ones.
    """
    # Identify column types automatically FROM THE ENGINEERED DATA
    numeric_features = [c for c in X_train.columns if c not in ['purpose', cfg.TARGET]]
    categorical_features = ['purpose']

    # Numeric Pipeline: Impute -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline: One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False # Keeps names clean (e.g., 'int.rate' instead of 'num__int.rate')
    )
    
    return preprocessor

def load_data():
    df = pd.read_csv(cfg.DATA_PATH)
    
    # Clean column names
    #df.columns = [c.replace('.', '_') for c in df.columns]
    
    # Apply Engineering HERE (Before Split)
    df = engineer_financial_features(df)
    
    X = df.drop(columns=[cfg.TARGET])
    y = df[cfg.TARGET]
    
    return X, y