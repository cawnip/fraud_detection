import logging
import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import CV_FOLDS, RANDOM_STATE, SCALE_POS_WEIGHT

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def tune_xgboost(X_train, y_train, n_trials: int = 30) -> dict:
    """
    Optuna ile XGBoost hyperparameter tuning.
    PR-AUC maximize edilir (imbalanced veri için).
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "scale_pos_weight": SCALE_POS_WEIGHT,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="average_precision", n_jobs=-1)
        return scores.mean()

    logger.info(f"Optuna tuning başlıyor ({n_trials} trial)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    })

    logger.info(f"En iyi PR-AUC: {study.best_value:.4f}")
    logger.info(f"En iyi parametreler: {best_params}")

    print(f"\n--- Optuna Sonucu ---")
    print(f"En iyi PR-AUC : {study.best_value:.4f}")
    print(f"En iyi params : {study.best_params}")

    return best_params


def build_tuned_xgboost(best_params: dict) -> XGBClassifier:
    return XGBClassifier(**best_params)
