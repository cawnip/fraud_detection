import joblib
import pandas as pd
from sklearn.metrics import classification_report
from utils.metrics import compute_metrics, find_optimal_threshold


def evaluate_model(model, X_test, y_test, model_name: str = "model") -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics['model'] = model_name

    print(f"\n=== {model_name} (threshold=0.5) ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"MCC       : {metrics['mcc']:.4f}")

    # Threshold optimizasyonu (F2 — recall ağırlıklı)
    opt = find_optimal_threshold(y_test, y_prob, beta=2.0)
    print(f"\n--- Optimal Threshold (F2, beta=2) ---")
    print(f"Threshold : {opt['threshold']:.4f}")
    print(f"Precision : {opt['precision']:.4f}")
    print(f"Recall    : {opt['recall']:.4f}")
    print(f"F2        : {opt['fbeta']:.4f}")
    print(f"MCC       : {opt['mcc']:.4f}")

    metrics['opt_threshold'] = opt['threshold']
    metrics['opt_recall'] = opt['recall']
    metrics['opt_precision'] = opt['precision']
    metrics['opt_f2'] = opt['fbeta']

    return metrics


def evaluate_all(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    results = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)

    df = pd.DataFrame(results).set_index('model')
    print("\n--- Model Karşılaştırması ---")
    cols = ['roc_auc', 'pr_auc', 'f1', 'mcc', 'opt_threshold', 'opt_recall', 'opt_precision', 'opt_f2']
    print(df[cols].to_string())
    return df


def save_threshold(model_name: str, threshold: float, save_dir: str = "models"):
    path = f"{save_dir}/{model_name}_threshold.joblib"
    joblib.dump(threshold, path)
    print(f"Threshold kaydedildi: {path} ({threshold:.4f})")
