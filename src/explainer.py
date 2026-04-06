import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)


def compute_shap(model, X_test: pd.DataFrame, max_samples: int = 500):
    """
    SHAP değerlerini hesaplar. Tree modeller için TreeExplainer kullanır.
    """
    logger.info("SHAP değerleri hesaplanıyor...")

    X_sample = X_test.iloc[:max_samples]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values, X_sample: pd.DataFrame, save_path: str = None):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_shap_bar(shap_values, X_sample: pd.DataFrame, save_path: str = None):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def explain_single(explainer, X_sample: pd.DataFrame, idx: int = 0):
    """Tek bir işlem için SHAP waterfall plot."""
    shap_values = explainer.shap_values(X_sample.iloc[[idx]])
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_sample.iloc[idx],
            feature_names=list(X_sample.columns),
        )
    )
