from src.data.load_data import import_and_load_data, split_cleaned_data
from src.features.build_features import preprocess_data, numerical_standartization
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
X_train_clean, X_test_clean, y_train_clean, y_test_clean = split_cleaned_data(clean_data, target_column_name)
X_train_clean_std, X_test_clean_std = numerical_standartization(X_train_clean, X_test_clean)

params = {}
model, y_pred_proba = xgbclf(params, X_train_clean, y_train_clean, X_test_clean, y_test_clean)
get_roc(y_test_clean, y_pred_proba)
plot_featureImportance(model, X_train_clean.columns)
