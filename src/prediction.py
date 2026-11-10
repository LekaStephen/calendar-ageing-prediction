from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/calendar_ageing_svr.joblib"
META_PATH = ROOT / "models/model_metadata.json"

def load_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model artefacts are missing. Run: python src/train_model.py")
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())

def predict_capacity_loss(model, temperature_c: float, soc_pct: float, storage_time_hours: float) -> dict:
    vals = [temperature_c, soc_pct, storage_time_hours]
    if not all(np.isfinite(vals)):
        raise ValueError("All inputs must be finite numbers")
    if not 0 <= temperature_c <= 60:
        raise ValueError("Temperature must be between 0 and 60 °C")
    if not 0 <= soc_pct <= 100:
        raise ValueError("SOC must be between 0 and 100%")
    if not 0 <= storage_time_hours <= 21241:
        raise ValueError("Storage time must be between 0 and 21,241 hours")
    X = pd.DataFrame([{
        "storage_time_days": storage_time_hours / 24.0,
        "temperature_c": temperature_c,
        "soc_pct": soc_pct,
    }])
    loss = float(model.predict(X)[0])
    return {"capacity_loss_pct": loss, "capacity_retention_pct": 100.0 - loss}
