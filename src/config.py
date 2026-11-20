from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = MODELS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "paysim_fraud_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

TARGET = "isFraud"
SEED = 42
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FPR_BUDGETS = [0.0001, 0.0005, 0.001]
LARGE_TRANSFER = 200_000
