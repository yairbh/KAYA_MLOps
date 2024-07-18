"""
Functions for preprocessing data, including converting target to binary, removing outliers,
standardizing numerical features, and creating dummy variables for categorical features.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def convert_target_to_binary(target, yes=None, no=None):
    # Handle binary representations set by the user
    if yes is not None and no is not None:
        mapping = {yes: 1, no: 0}

    # Handle common binary representations
    elif target.dtype == 'object':
        mapping = {
            'yes': 1, 'no': 0,
            'true': 1, 'false': 0,
            'True': 1, 'False': 0,
            'Yes': 1, 'No': 0
        }
    else: mapping = None

    if mapping:
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
