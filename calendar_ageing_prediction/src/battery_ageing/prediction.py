from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_PATH, FEATURE_NAMES

class ModelArtifactError(RuntimeError):
    pass

def load_model(model_path=MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise ModelArtifactError("Trained RBF-SVR model is missing. Audit the source data and train the model before predicting.")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ModelArtifactError(f"Could not load model artefact: {exc}") from exc

def predict_capacity_metrics(temperature_c, soc_percent, duration_hours, model_path=MODEL_PATH):
    model = load_model(model_path)
    row = pd.DataFrame([{
        "temperature_c": float(temperature_c),
        "soc_percent": float(soc_percent),
        "storage_duration_days": float(duration_hours) / 24.0,
    }], columns=FEATURE_NAMES)
    loss = float(np.asarray(model.predict(row)).reshape(-1)[0])
    if not np.isfinite(loss):
        raise ModelArtifactError("Model returned a non-finite prediction.")
    return {"capacity_loss_percent": loss, "capacity_retention_percent": 100.0 - loss}
