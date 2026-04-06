import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_class_distribution(df: pd.DataFrame, save_path: str = None):
    counts = df['Class'].value_counts()
    labels = ['Normal', 'Fraud']
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=counts.values, hue=labels, palette=['steelblue', 'tomato'], legend=False)
    plt.title('Class Dağılımı')
    plt.ylabel('İşlem Sayısı')
    for i, v in enumerate(counts.values):
        plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_amount_distribution(df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label, color in zip(axes, [0, 1], ['steelblue', 'tomato']):
        data = df[df['Class'] == label]['Amount']
        ax.hist(data, bins=50, color=color, edgecolor='white')
        ax.set_title(f"{'Normal' if label == 0 else 'Fraud'} - Amount Dağılımı")
        ax.set_xlabel('Amount')
        ax.set_ylabel('Frekans')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(16, 12))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                linewidths=0.5, annot=False)
    plt.title('Feature Korelasyon Matrisi')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name: str = "", save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name: str = "", save_path: str = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='tomato', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Eğrisi — {model_name}')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15, save_path: str = None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    feat_labels = [feature_names[i] for i in indices]
    sns.barplot(x=importances[indices], y=feat_labels, hue=feat_labels, palette='viridis', legend=False)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
