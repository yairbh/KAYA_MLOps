from src.data.load_data import import_and_load_data, split_cleaned_data
from src.features.build_features import preprocess_data, numerical_standartization
from src.features.SMOTE import show_distribution_of_labels, smote
from src.models.train_model import xgbclf
from src.visualization.visualize import get_roc, plot_featureImportance

data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

column_names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
                'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
                'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
                'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

target_column_name = 'classification'

data = import_and_load_data(data_url, column_names)
clean_data = preprocess_data(data, target_column_name)

# Run on clean data (imbalanced)
X_train_clean, X_test_clean, y_train_clean, y_test_clean = split_cleaned_data(clean_data, target_column_name)
X_train_clean_std, X_test_clean_std = numerical_standartization(X_train_clean, X_test_clean)

params = {}  # Use default parameters
params1={
    'n_estimators':3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'gamma':0.1,
    'subsample':0.8,
    'colsample_bytree':0.3,
    'min_child_weight':3,
    'max_depth':3,
    #'seed':1024,
    'n_jobs' : -1
}

params2={
    'n_estimators':3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.005,
    #'gamma':0.01,
    'subsample':0.555,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'max_depth':8,
    #'seed':1024,
    'n_jobs' : -1
}

model, y_pred_proba = xgbclf(params2, X_train_clean_std, y_train_clean, X_test_clean_std, y_test_clean)

get_roc(y_test_clean, y_pred_proba)
plot_featureImportance(model, X_train_clean_std.columns)

# Run on oversampling data (balanced)
X_train_oversampled, y_train_oversampled, X_test_smote, y_test_smote = smote(clean_data,target_column_name) #the last two variables are just made from the different split in smote func

X_train_oversampled_std, X_test_oversampled_std = numerical_standartization(X_train_oversampled, X_test_smote)

model_smote, y_pred_proba_smote = xgbclf(params2, X_train_oversampled_std, y_train_oversampled, X_test_oversampled_std, y_test_smote)

get_roc(y_test_smote, y_pred_proba_smote)
plot_featureImportance(model_smote, X_train_oversampled_std.columns, title='Feature Importance when oversampling')
