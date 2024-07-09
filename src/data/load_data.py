import pandas as pd
from sklearn.model_selection import train_test_split

def import_and_load_data(data_url, column_names):
    data = pd.read_csv(data_url, sep=' ', header=None, names=column_names)
    return data

def split_cleaned_data(clean_data: pd.DataFrame, target_column_name):
    """
    return: X_train_clean, X_test_clean, y_train_clean, y_test_clean
    """
    from sklearn.model_selection import train_test_split

    # Unscaled, unnormalized data
    X_clean = clean_data.drop(target_column_name, axis=1)
    y_clean = clean_data[target_column_name]
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2,
                                                                                random_state=1)
    return X_train_clean, X_test_clean, y_train_clean, y_test_clean
