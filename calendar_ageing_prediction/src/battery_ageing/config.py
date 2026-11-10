from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "rbf_svr_pipeline.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

APP_TITLE = "Li-ion Battery Calendar Aging Prediction"
FEATURE_NAMES = ["temperature_c", "soc_percent", "storage_duration_days"]
MIN_TEMPERATURE_C, MAX_TEMPERATURE_C = 0.0, 60.0
MIN_SOC_PERCENT, MAX_SOC_PERCENT = 0.0, 100.0
MIN_DURATION_HOURS, MAX_DURATION_HOURS = 0.0, 21241.0
