import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler
from collections import defaultdict
import xgboost as xgb
from xgboost import XGBClassifier
import seaborn
from pprint import pprint


def import_and_load_data(data_url, column_names):
    data = pd.read_csv(data_url, sep=' ', header=None, names=column_names)
    return data


def convert_target_to_binary(target):
    # Handle common binary representations
    if target.dtype == 'object':
        mapping = {
            'yes': 1, 'no': 0,
            'true': 1, 'false': 0,
            'True': 1, 'False': 0,
            'Yes': 1, 'No': 0
        }
        target = target.map(mapping).fillna(target)
    unique_values = target.dropna().unique()
    if len(unique_values) != 2:
        raise ValueError("Target column must have exactly two unique values.")

    # Custom mapping rule: if 1 is present, map 1 to 1 and the other value to 0
    if 1 in unique_values:
        return target.map({1: 1, unique_values[unique_values != 1][0]: 0})
    # General case: sort values and map the lower to 0 and higher to 1
    else:
        unique_values = sorted(unique_values)
        return target.map({unique_values[0]: 0, unique_values[1]: 1})


def remove_outliers(column):
    mean = column.mean()
    std = column.std()
    return column.mask(((column - mean).abs() > 3 * std), mean)


def numerical_standartization(X_train_clean, X_test_clean):
    numerical_columns = []
    for col in X_train_clean.columns:
        unique_values = X_train_clean[col].dropna().unique()
        if len(unique_values) == 2 and set(unique_values) == {0, 1}:
            continue  # Skip boolean columns
        elif X_train_clean[col].dtype in ['int64', 'float64']:
            numerical_columns.append(col)

    scaler = StandardScaler()
    X_train_clean[numerical_columns] = scaler.fit_transform(X_train_clean[numerical_columns])
    X_test_clean[numerical_columns] = scaler.transform(X_test_clean[numerical_columns])

    return X_train_clean, X_test_clean


def preprocess_data(data: pd.DataFrame, target_column_name):
    # Step 1: Convert target values to binary (0/1)
    data[target_column_name] = convert_target_to_binary(data[target_column_name])

    # Step 2: Identify categorical columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column_name in categorical_columns:
        categorical_columns.remove(target_column_name)

    # Step 3: Identify numerical columns
    numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
    if target_column_name in numerical_columns:
        numerical_columns.remove(target_column_name)

    # Step 4: Convert categorical columns into dummy variables
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Step 5: Convert boolean columns to 0/1
    boolean_columns = data.select_dtypes(include=['bool']).columns.tolist()
    data[boolean_columns] = data[boolean_columns].astype(int)

    # Step 6: Remove outliers from the numerical columns
    numerical_columns = [col for col in numerical_columns if col not in boolean_columns]
    data[numerical_columns] = data[numerical_columns].apply(remove_outliers)

    return data


def split_cleaned_data(clean_data: pd.DataFrame, target_column_name):
    """
    return: X_train_clean, X_test_clean, y_train_clean, y_test_clean
    """
    # Unscaled, unnormalized data
    X_clean = clean_data.drop(target_column_name, axis=1)
    y_clean = clean_data[target_column_name]
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2,
                                                                                random_state=1)
    return X_train_clean, X_test_clean, y_train_clean, y_test_clean


def get_eval1(clf, X, y):
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(clf, X, y, cv=2, scoring='accuracy')
    scores2 = cross_val_score(clf, X, y, cv=2, scoring='precision')
    scores3 = cross_val_score(clf, X, y, cv=2, scoring='recall')
    scores4 = cross_val_score(clf, X, y, cv=2, scoring='roc_auc')

    # The mean score and standard deviation of the score estimate
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
    print("Cross Validation Precision: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
    print("Cross Validation Recall: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
    print("Cross Validation roc_auc: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std()))

    return


def get_eval2(clf, X_train, y_train, X_test, y_test):
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(clf, X_test, y_test, cv=2, scoring='accuracy')
    scores2 = cross_val_score(clf, X_test, y_test, cv=2, scoring='precision')
    scores3 = cross_val_score(clf, X_test, y_test, cv=2, scoring='recall')
    scores4 = cross_val_score(clf, X_test, y_test, cv=2, scoring='roc_auc')

    # The mean score and standard deviation of the score estimate
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
    print("Cross Validation Precision: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
    print("Cross Validation Recall: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
    print("Cross Validation roc_auc: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std()))

    return


# Function to get roc curve
def get_roc(y_test, y_pred):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # Plot of a ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="upper left")
    plt.show()
    return


def xgbclf(params, X_train, y_train, X_test, y_test):
    eval_set = [(X_train, y_train), (X_test, y_test)]

    model = XGBClassifier(**params).fit(X_train, y_train, eval_set=eval_set, eval_metric='auc',
                                        early_stopping_rounds=100, verbose=100)

    best_iteration = model.best_iteration

    # Set n_estimators directly (optional)
    if best_iteration is not None:
        model.set_params(n_estimators=best_iteration)

    # Train again with the best number of trees (optional)
    model.fit(X_train, y_train)

    # Predict target variables y for test data
    y_pred = model.predict(X_test)

    # Create and print confusion matrix
    abclf_cm = confusion_matrix(y_test, y_pred)
    print(abclf_cm)

    # y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('\n')
    print("Model Final Generalization Accuracy: %.6f" % accuracy_score(y_test, y_pred))

    # Predict probabilities target variables y for test data
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # model.best_iteration
    get_roc(y_test, y_pred_proba)
    return model


def plot_featureImportance(model, keys):
    importances = model.feature_importances_

    importance_frame = pd.DataFrame({'Importance': list(importances), 'Feature': list(keys)})
    importance_frame.sort_values(by='Importance', inplace=True)
    importance_frame.tail(10).plot(kind='barh', x='Feature', figsize=(8, 8), color='orange')


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
model = xgbclf(params, X_train_clean, y_train_clean, X_test_clean, y_test_clean)

