"""
Main script for loading data, preprocessing, training models, applying SHAP for feature selection,
and evaluating performance with and without SMOTE.
"""

import pandas as pd
from src.data.load_data import import_and_load_data, split_cleaned_data
from src.features.build_features import preprocess_data, numerical_standartization
from src.features.SMOTE import show_distribution_of_labels, smote
from src.models.train_model import xgbclf
from src.visualization.visualize import get_roc, plot_featureImportance
from src.features.SHAP import shap_filter
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 1. Data Loading
data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
                'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
                'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
                'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']
target_column_name = 'classification'
data = import_and_load_data(data_url, column_names)

# 2. Data Preprocessing
clean_data = preprocess_data(data, target_column_name)

# 3. Data Splitting
X_train_clean, X_test_clean, y_train_clean, y_test_clean = split_cleaned_data(clean_data, target_column_name)

# 4. Data Standardization
X_train_clean_std, X_test_clean_std = numerical_standartization(X_train_clean, X_test_clean)

params = {}  # Use default parameters
params1 = {
    'n_estimators': 3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.3,
    'min_child_weight': 3,
    'max_depth': 3,
    'n_jobs': -1
}

params2 = {
    'n_estimators': 3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.005,
    'subsample': 0.555,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'max_depth': 8,
    'n_jobs': -1
}

# Control whether to generate plots
generate_plots = False

# 5. Initial Model Training (Imbalanced Data)
print("\nInitial Model (Imbalanced Data):")
initial_model, y_pred_proba = xgbclf(params2, X_train_clean_std, y_train_clean, X_test_clean_std, y_test_clean)
initial_auc = roc_auc_score(y_test_clean, y_pred_proba)
get_roc(y_test_clean, y_pred_proba, title="ROC Curve - Initial Model (Imbalanced Data)")
if generate_plots:
    plot_featureImportance(initial_model, X_train_clean.columns)

# 6. SHAP Feature Selection (Imbalanced Data)
features_to_remove = shap_filter(pd.DataFrame(X_train_clean_std, columns=X_train_clean.columns), initial_model, importance_threshold=0.2)
X_train_shap = pd.DataFrame(X_train_clean_std, columns=X_train_clean.columns).drop(columns=features_to_remove)
X_test_shap = pd.DataFrame(X_test_clean_std, columns=X_test_clean.columns).drop(columns=features_to_remove)

# 7. Retrain Model with Selected Features (Imbalanced Data)
print("\nModel after SHAP Feature Selection (Imbalanced Data):")
model_shap, y_pred_proba_shap = xgbclf(params2, X_train_shap.values, y_train_clean, X_test_shap.values, y_test_clean)
shap_auc = roc_auc_score(y_test_clean, y_pred_proba_shap)
get_roc(y_test_clean, y_pred_proba_shap, title="ROC Curve - SHAP Feature Selection (Imbalanced Data)")
if generate_plots:
    plot_featureImportance(model_shap, X_train_shap.columns)

# 8. Data Balancing with SMOTE
X_train_oversampled, y_train_oversampled, X_test_oversampled, y_test_oversampled = smote(clean_data, target_column_name)
X_train_oversampled_std, X_test_oversampled_std = numerical_standartization(X_train_oversampled, X_test_oversampled)

# 9. Model Training on Oversampled Data (Balanced Data)
print("\nModel after SMOTE (Balanced Data):")
model_smote, y_pred_proba_smote = xgbclf(params2, X_train_oversampled_std, y_train_oversampled, X_test_oversampled_std, y_test_oversampled)
smote_auc = roc_auc_score(y_test_oversampled, y_pred_proba_smote)
get_roc(y_test_oversampled, y_pred_proba_smote, title="ROC Curve - SMOTE (Balanced Data)")
if generate_plots:
    plot_featureImportance(model_smote, X_train_oversampled.columns, title='Feature Importance after SMOTE')

# 10. SHAP Feature Selection on Oversampled Data (Balanced Data)
X_train_oversampled_df = pd.DataFrame(X_train_oversampled_std, columns=X_train_oversampled.columns)
X_test_oversampled_df = pd.DataFrame(X_test_oversampled_std, columns=X_test_oversampled.columns)
features_to_remove_oversampled = shap_filter(X_train_oversampled_df, model_smote, importance_threshold=0.2)
X_train_oversampled_shap = X_train_oversampled_df.drop(columns=features_to_remove_oversampled)
X_test_oversampled_shap = X_test_oversampled_df.drop(columns=features_to_remove_oversampled)

# 11. Retrain Model with Selected Features on Oversampled Data (Balanced Data)
print("\nModel after SHAP Feature Selection (Balanced Data):")
model_smote_shap, y_pred_proba_smote_shap = xgbclf(params2, X_train_oversampled_shap.values, y_train_oversampled, X_test_oversampled_shap.values, y_test_oversampled)
smote_shap_auc = roc_auc_score(y_test_oversampled, y_pred_proba_smote_shap)
get_roc(y_test_oversampled, y_pred_proba_smote_shap, title="ROC Curve - SHAP Feature Selection (Balanced Data)")
if generate_plots:
    plot_featureImportance(model_smote_shap, X_train_oversampled_shap.columns, title='Feature Importance after SHAP and SMOTE')

# 12. Summary of Results
print("\nSummary of Results:")
print(f"Initial Model AUC (Imbalanced): {initial_auc:.4f}")
print(f"SHAP-filtered Model AUC (Imbalanced): {shap_auc:.4f}")
print(f"Model AUC after SMOTE (Balanced): {smote_auc:.4f}")
print(f"SHAP-filtered Model AUC after SMOTE (Balanced): {smote_shap_auc:.4f}")
