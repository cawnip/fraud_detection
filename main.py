import logging
import numpy as np
from config import LOG_DIR, USE_SMOTE, DEPLOY_MODEL, BETA

# ---------------------------------------------------------------------------
# Logging kurulumu
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "training.log"),
    ],
)
logger = logging.getLogger(__name__)

from src.data_loader import load_data, get_basic_info
from src.preprocessing import preprocess
from src.model import get_models
from src.trainer import train_all
from src.evaluator import evaluate_all, save_threshold
from src.cross_validation import cross_validate
from src.tuner import tune_xgboost, build_tuned_xgboost
from src.explainer import compute_shap, plot_shap_summary, plot_shap_bar
from src.visualization import (
    plot_class_distribution,
    plot_amount_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)
from utils.metrics import find_optimal_threshold


def main():
    logger.info("=== Fraud Detection Pipeline Başlıyor ===")

    # 1. Veri yükleme
    df = load_data()
    get_basic_info(df)

    # 2. EDA görselleri
    plot_class_distribution(df)
    plot_amount_distribution(df)

    # 3. Ön işleme
    X_train, X_test, y_train, y_test = preprocess(df, use_smote=USE_SMOTE)

    # 4. Hyperparameter tuning (XGBoost)
    logger.info("Hyperparameter tuning başlıyor...")
    best_params = tune_xgboost(X_train.values, y_train.values, n_trials=30)
    tuned_xgb = build_tuned_xgboost(best_params)

    # 5. Tüm modelleri al ve tuned XGBoost ile değiştir
    models = get_models()
    models["xgboost"] = tuned_xgb

    # 6. Cross Validation (XGBoost üzerinde)
    logger.info("Cross Validation başlıyor...")
    cross_validate(tuned_xgb, X_train.values, y_train.values)

    # 7. Model eğitimi
    trained_models = train_all(models, X_train, y_train)

    # 8. Değerlendirme
    results = evaluate_all(trained_models, X_test, y_test)

    # 9. En iyi model
    best_model_name = results['roc_auc'].idxmax()
    best_model = trained_models[best_model_name]
    logger.info(f"En iyi model: {best_model_name}")
    print(f"\nEn iyi model: {best_model_name}")

    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 10. Threshold optimizasyonu & kaydet
    opt = find_optimal_threshold(y_test, y_prob, beta=BETA)
    save_threshold(best_model_name, opt['threshold'])

    # 11. Görseller
    y_pred_opt = (y_prob >= opt['threshold']).astype(int)
    plot_confusion_matrix(
        y_test, y_pred_opt,
        model_name=f"{best_model_name} (threshold={opt['threshold']:.2f})"
    )
    plot_roc_curve(y_test, y_prob, model_name=best_model_name)

    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, list(X_test.columns))

    # 12. SHAP
    logger.info("SHAP hesaplanıyor...")
    explainer, shap_values, X_sample = compute_shap(best_model, X_test)
    plot_shap_summary(shap_values, X_sample)
    plot_shap_bar(shap_values, X_sample)

    logger.info("=== Pipeline Tamamlandı ===")


if __name__ == "__main__":
    main()
