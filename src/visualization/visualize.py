"""
Visualization functions for plotting ROC curves and feature importance.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd


def get_roc(y_test, y_pred, title='Receiver operating characteristic'):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="upper left")
    plt.show()
    return


def plot_featureImportance(model, feature_names, title='Feature Importance'):
    importances = model.feature_importances_
    importance_frame = pd.DataFrame({'Importance': importances, 'Feature': feature_names})
    importance_frame.sort_values(by='Importance', inplace=True)
    importance_frame.tail(10).plot(kind='barh', x='Feature', figsize=(8, 8), color='orange')
    plt.xlabel('Importance')
    plt.title(title)
    plt.show()
