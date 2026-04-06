import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import TEST_SIZE, RANDOM_STATE

logger = logging.getLogger(__name__)


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])
    logger.info("Amount ve Time ölçeklendirildi.")
    return df


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = RANDOM_STATE):
    logger.info(f"SMOTE öncesi: {dict(pd.Series(y_train).value_counts())}")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE sonrası: {dict(pd.Series(y_res).value_counts())}")
    print(f"SMOTE öncesi: {dict(pd.Series(y_train).value_counts())}")
    print(f"SMOTE sonrası: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def preprocess(df: pd.DataFrame, use_smote: bool = True):
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    return X_train, X_test, y_train, y_test
