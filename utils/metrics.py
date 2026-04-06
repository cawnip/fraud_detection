import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def find_optimal_threshold(y_true, y_prob, beta: float = 2.0) -> dict:
    """
    PR eğrisi üzerinden optimal threshold bulur.
    beta > 1 → recall ağırlıklı (fraud detection için önerilir)
    beta = 1 → F1 (precision/recall dengeli)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    beta2 = beta ** 2
    fbeta = (
        (1 + beta2) * (precisions * recalls)
        / (beta2 * precisions + recalls + 1e-8)
    )

    best_idx = np.argmax(fbeta[:-1])
    best_threshold = thresholds[best_idx]

    y_pred_opt = (y_prob >= best_threshold).astype(int)

    return {
        "threshold": best_threshold,
        "precision": precision_score(y_true, y_pred_opt, zero_division=0),
        "recall": recall_score(y_true, y_pred_opt, zero_division=0),
        "f1": f1_score(y_true, y_pred_opt, zero_division=0),
        "fbeta": fbeta[best_idx],
        "mcc": matthews_corrcoef(y_true, y_pred_opt),
    }
