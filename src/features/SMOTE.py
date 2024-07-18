"""
Functions for applying SMOTE to handle imbalanced data and visualizing label distributions.
"""

from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
from src.data.load_data import split_cleaned_data

def show_distribution_of_labels(data_target: pd.Series, string = 'Distribution of Labels'):
    label_counts = data_target.value_counts()
    labels = label_counts.index
    sizes = label_counts.values
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  # explode the 1st slice (label 0)

    plt.figure(figsize=(4, 4))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title(string)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    return


# For imbalanced data. DONE ONLY on the train. so this comes before the split
def smote(data: pd.DataFrame,target_column_name, random_state = 1):
    y_clean_all = data[target_column_name]
    X_clean_all = data.drop(target_column_name, axis = 1)
    show_distribution_of_labels(y_clean_all, string = 'Distribution of All Labels Before SMOTE')
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = split_cleaned_data(data, target_column_name)
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train_clean, y_train_clean)
    show_distribution_of_labels(y_resampled,string = 'Distribution of Train Labels After SMOTE')
    return X_resampled, y_resampled, X_test_clean, y_test_clean
