from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import pandas as pd

def train_logistic_regression(X_train, y_train):
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Done.")
    return model

def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Done.")
    return model

def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    print("Done.")
    return model

def evaluate(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    return y_pred, y_prob