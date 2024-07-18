import shap
import pandas as pd

def shap_filter(clean_data, model, importance_threshold=0.2):
    """
       This function receives the clean data (excluding the target column) and the trained model,
       and returns a list of the least 20% important features. Threshold can be adjusted.
    """
    print(f"Shape of clean_data: {clean_data.shape}")
    explainer = shap.TreeExplainer(model)  # This is the trained model
    shap_values = explainer.shap_values(clean_data)  # SHAP values for train dataset

    # Calculate mean absolute SHAP values
    mean_abs_shap_values = pd.DataFrame(abs(shap_values), columns=clean_data.columns).mean()

    # Sort the mean absolute SHAP values to find the feature importance
    mean_abs_shap_values_sorted = mean_abs_shap_values.sort_values(ascending=True)

    feature_importance = pd.Series(mean_abs_shap_values, index=clean_data.columns)

    # Determine the number of features to remove ({importance_threshold}% of the total number of features)
    num_features_to_remove = int(len(feature_importance) * importance_threshold)

    # Identify the least important features
    features_to_remove = feature_importance.nsmallest(num_features_to_remove).index.tolist()

    return features_to_remove
