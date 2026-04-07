import logging
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import TEST_SIZE, RANDOM_STATE, MODEL_DIR

logger = logging.getLogger(__name__)

SCALE_COLS = ['Time', 'Amount']


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Scaler'ı yalnızca X_train üzerinde fit eder, X_test'e transform uygular.
    Kaydedilen scaler inference sırasında app.py tarafından kullanılır.
    """
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])

    MODEL_DIR.mkdir(exist_ok=True)
    scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler kaydedildi: {scaler_path}")

    return X_train, X_test


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
    return X_res, y_res


def preprocess(df: pd.DataFrame, use_smote: bool = True):
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = scale_features(X_train, X_test)
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    return X_train, X_test, y_train, y_test
