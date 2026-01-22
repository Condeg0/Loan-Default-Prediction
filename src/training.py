from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import src.config as cfg
import src.data_pipeline as dp
from scipy.stats import loguniform, uniform


def train_benchmarks(X_train, y_train):
    """
    Trains multiple models to find the best baseline.
    Returns a dictionary of trained pipelines.
    """
    # Get the preprocessing logic
    preprocessor = dp.get_preprocessing_pipeline(X_train)
    
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=300, n_jobs=-1),
        'XGBoost': XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), eval_metric='logloss')
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        # Create full pipeline: Raw Data -> Preprocessing -> Model
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
       
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        
    return trained_models

def tune(X_train, y_train, model_name):
    """
    Fine-tunes the best performing model (usually XGBoost) for Accuracy/AUC.
    """
    preprocessor = dp.get_preprocessing_pipeline(X_train)
    0v
    
    # Define grid for RandomizedSearch & initialize model
    if model_name == "XGBoost":
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 4, 5, 6],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.7, 0.8, 0.9],
            'classifier__colsample_bytree': [0.7, 0.8, 0.9],
            # Heavily penalize errors on the minority class (Defaults)
            'classifier__scale_pos_weight': [1, 5, 10] 
        }
        model = XGBClassifier()

    elif model_name == "RandomForest":
        param_grid = {
            'classifier__n_estimators': [200, 300, 400],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

    elif model_name == "LogisticRegression":
        param_grid = [
            {
                'classifier__solver': ['liblinear'],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__C': loguniform(1e-4, 1e2)  
            },
            {
                'classifier__solver': ['lbfgs'],
                'classifier__penalty': ['l2'],
                'classifier__C': loguniform(1e-4, 1e2)
            },
            {
                'classifier__solver': ['saga'],
                'classifier__penalty': ['elasticnet'],
                'classifier__C': loguniform(1e-4, 1e2),
                'classifier__l1_ratio': uniform(0, 1)   
            }
        ]
        model = LogisticRegression(class_weight='balanced', max_iter=5000)
        #param_grid = {
            #'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            #'classifier__solver': ['liblinear', 'saga'],
        #}
        model = LogisticRegression(class_weight='balanced')
    

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    print(f'Tuning {model_name}', end=' --> ')
    search = RandomizedSearchCV(
        pipeline, 
        param_grid, 
        n_iter=35, 
        scoring='roc_auc', # Optimize for separation power
        cv=3,
        verbose=1,
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_
