from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=577,  # ~non_fraud / fraud oranı
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ),
    }
