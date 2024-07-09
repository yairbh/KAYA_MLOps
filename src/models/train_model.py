import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.visualization.visualize import get_roc

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
    return model, y_pred_proba
