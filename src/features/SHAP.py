#!pip install shap   #add to requirments
import shap
import pandas as pd


def shap_filter(clean_data,  model, importance_threshold = 0.2):
    """
       this function receive the clean data (don't need the target column) and the trained model,
       and returns a list of the least 20% important features. threshold can be adjusted
    """

    explainer = shap.TreeExplainer(model)                       # this is the trained model
    shap_values = explainer.shap_values(clean_data)             # SHAP values for train dataset

    # Calculate mean absolute SHAP values
    mean_abs_shap_values = pd.DataFrame(abs(shap_values), columns=clean_data.columns).mean()

    # Sort the mean absolute SHAP values to find the feature important
    mean_abs_shap_values_sorted = mean_abs_shap_values.sort_values(ascending=True)

    feature_importance = pd.Series(mean_abs_shap_values, index=clean_data.columns)

    # Determine the number of features to remove ({importance_threshold}% of the total number of features)
    num_features_to_remove = int(len(feature_importance) * importance_threshold)

    # Identify the least important features
    features_to_remove = feature_importance.nsmallest(num_features_to_remove).index.tolist()

    return features_to_remove



def shap_analysis(train_data, test_data, model):
    """
       this function receive the train set, test set and the trained model, and returns some SHAP analysis.
    """

    # Create a SHAP explainer with a background dataset (e.g., a subset of the training data)
    background_data = shap.sample(train_data, 100)
    explainer = shap.Explainer(model, background_data)

    # Compute SHAP values for the test set
    shap_values = explainer(test_data)

    # Plot SHAP feature importance
    print("\nSHAP feature importance:\n")
    shap.plots.bar(shap_values)

    # Calculate mean absolute SHAP values
    mean_abs_shap_values = pd.DataFrame(abs(shap_values.values), columns=X_test_clean.columns).mean()

    # Sort the mean absolute SHAP values to find the least important feature
    mean_abs_shap_values_sorted = mean_abs_shap_values.sort_values(ascending=True)

    # Print the sorted mean absolute SHAP values
    print("\nComplete feature importance (ascending):")
    print(mean_abs_shap_values_sorted)

    # plot importance bar plot
    print("\nSHAP values density plot:\nIdentify how much impact each feature has on the model.\nFeatures are sorted by the sum of their SHAP value magnitudes across all samples.\n")
    shap.plots.beeswarm(shap_values)

    print("\nBeeswarm plot:\nThis is an absolute value of the SHAP values density plot.\n(the bar plots above are the summary statistics from the values shown in the beeswarm plot)")
    shap.plots.beeswarm(shap_values.abs, color="shap_red")

    #plot heatmap
    print(f"\nSHAP heatmap:\n")
    shap.plots.heatmap(shap_values[:1000])

    return
