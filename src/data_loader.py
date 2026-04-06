import logging
import pandas as pd
import os
from config import DATA_PATH

logger = logging.getLogger(__name__)


def load_data(path: str = None) -> pd.DataFrame:
    path = path or str(DATA_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset bulunamadı: {path}")
    df = pd.read_csv(path)
    logger.info(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


def get_basic_info(df: pd.DataFrame) -> None:
    fraud_ratio = df['Class'].mean() * 100
    logger.info(f"Fraud oranı: %{fraud_ratio:.4f}")
    print("\n--- Temel Bilgiler ---")
    print(f"Boyut: {df.shape}")
    print(f"Eksik değer: {df.isnull().sum().sum()}")
    print(f"\nClass dağılımı:\n{df['Class'].value_counts()}")
    print(f"Fraud oranı: %{fraud_ratio:.4f}")
