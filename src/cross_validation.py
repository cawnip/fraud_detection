import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from utils.metrics import compute_metrics
from config import CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)


def cross_validate(model, X, y, n_splits: int = CV_FOLDS) -> pd.DataFrame:
    """
    Stratified K-Fold CV uygular.
    Her fold'da fraud oranı korunur.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    logger.info(f"Stratified {n_splits}-Fold CV başlıyor...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_val)
        y_prob = fold_model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_pred, y_prob)
        metrics['fold'] = fold
        fold_results.append(metrics)

        logger.info(
            f"Fold {fold}/{n_splits} — "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

    df = pd.DataFrame(fold_results).set_index('fold')

    print("\n--- Cross Validation Sonuçları ---")
    print(df.to_string())
    print("\n--- Ortalama ± Std ---")
    for col in ['roc_auc', 'pr_auc', 'f1', 'mcc']:
        print(f"{col:12s}: {df[col].mean():.4f} ± {df[col].std():.4f}")

    return df
