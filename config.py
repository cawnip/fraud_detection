from pathlib import Path
import os

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
PLOT_DIR = ROOT_DIR / "plots"

DATA_PATH = DATA_DIR / "creditcard.csv"

# Preprocessing
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_SMOTE = True

# Models
SCALE_POS_WEIGHT = 577  # ~non_fraud / fraud oranı

# Training
CV_FOLDS = 5

# Threshold optimization
BETA = 2.0  # F-beta: beta > 1 → recall ağırlıklı

# Best model for deploy
DEPLOY_MODEL = "xgboost"

# App
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
