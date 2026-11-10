from pathlib import Path
import json, hashlib, sys
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib
from data_loader import load_capacity_workbook, FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source/DischargeCapacity.xlsx"
MODEL_DIR = ROOT / "models"
PROCESSED = ROOT / "data/processed/reconstructed_capacity_analysis.csv"

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_capacity_workbook(SOURCE)
    df.to_csv(PROCESSED, index=False)
    X, y, groups = df[FEATURES], df[TARGET], df.condition_id
    pipe = Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    grid = {
        "svr__C": [1, 3, 10, 30, 100],
        "svr__epsilon": [0.05, 0.15, 0.30, 0.50],
        "svr__gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
    }
    search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=GroupKFold(n_splits=3), n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    model_path = MODEL_DIR / "calendar_ageing_svr.joblib"
    joblib.dump(search.best_estimator_, model_path)
    sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    metadata = {
        "model": "RBF Support Vector Regression",
        "features": FEATURES,
        "target": TARGET,
        "target_definition": "100 * (initial_capacity_ah - capacity_ah) / initial_capacity_ah",
        "time_unit": "days",
        "temperature_unit": "degrees Celsius",
        "soc_unit": "percent",
        "selected_hyperparameters": search.best_params_,
        "inner_grouped_cv_folds": 3,
        "inner_selection_metric": "MAE (percentage points)",
        "best_inner_grouped_mae_percentage_points": float(-search.best_score_),
        "training_rows": int(len(df)),
        "condition_groups": int(groups.nunique()),
        "source_workbook_sha256": sha,
        "note": "Final full-development-data fit for application inference; nested grouped out-of-fold scores remain the evaluation evidence. Not cell-level or external validation."
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
