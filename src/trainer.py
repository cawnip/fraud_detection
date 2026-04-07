import logging
import joblib
import os
import time
from config import MODEL_DIR

logger = logging.getLogger(__name__)


def train_model(model, X_train, y_train, model_name: str = "model"):
    logger.info(f"[{model_name}] Eğitim başlıyor...")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info(f"[{model_name}] Eğitim tamamlandı ({elapsed:.1f}s)")
    return model


def save_model(model, model_name: str, save_dir: str = None):
    save_dir = save_dir or str(MODEL_DIR)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.joblib")
    joblib.dump(model, path)
    logger.info(f"Model kaydedildi: {path}")
    return path


def load_model(model_name: str, save_dir: str = None):
    save_dir = save_dir or str(MODEL_DIR)
    path = os.path.join(save_dir, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bulunamadı: {path}")
    logger.info(f"Model yüklendi: {path}")
    return joblib.load(path)


def train_all(models: dict, X_train, y_train, save_dir: str = None) -> dict:
    trained = {}
    for name, model in models.items():
        trained[name] = train_model(model, X_train, y_train, model_name=name)
        save_model(trained[name], name, save_dir)
    return trained
