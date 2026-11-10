from pathlib import Path
import re
import pandas as pd

FEATURES = ["storage_time_days", "temperature_c", "soc_pct"]
TARGET = "capacity_loss_pct"

def load_capacity_workbook(path: str | Path) -> pd.DataFrame:
    """Reconstruct condition-level capacity trajectories from the Naumann workbook."""
    raw = pd.read_excel(path, header=None)
    labels = raw.iloc[1, 1:].tolist()
    data = raw.iloc[2:, :].copy().dropna(subset=[0])
    times = pd.to_numeric(data[0], errors="raise").to_numpy(dtype=float)
    records = []
    for col, label in enumerate(labels, start=1):
        if not isinstance(label, str):
            continue
        temp_match = re.search(r"TP_(\d+)°C", label)
        soc_match = re.search(r",(\d+(?:\.\d+)?)%SOC", label)
        if not temp_match or not soc_match:
            continue
        temperature = float(temp_match.group(1))
        soc = float(soc_match.group(1))
        values = pd.to_numeric(data[col], errors="raise").to_numpy(dtype=float)
        initial = values[times.argmin()]
        condition_id = re.sub(r"_+", "_", label.replace("°C", "_C").replace("%SOC", "_SOC").replace(",", "_").replace(".", "_")).strip("_")
        for time_h, capacity in zip(times, values):
            records.append({
                "condition_id": condition_id,
                "condition_label": label,
                "temperature_c": temperature,
                "soc_pct": soc,
                "storage_time_hours": float(time_h),
                "storage_time_days": float(time_h) / 24.0,
                "capacity_ah": float(capacity),
                "initial_capacity_ah": float(initial),
            })
    result = pd.DataFrame(records)
    result[TARGET] = 100.0 * (result.initial_capacity_ah - result.capacity_ah) / result.initial_capacity_ah
    result["capacity_retention_pct"] = 100.0 - result[TARGET]
    result = result.sort_values(["condition_id", "storage_time_hours"]).reset_index(drop=True)
    if len(result) != 595 or result.condition_id.nunique() != 17:
        raise ValueError(f"Unexpected dataset structure: {len(result)} rows, {result.condition_id.nunique()} conditions")
    if result.duplicated(["condition_id", "storage_time_hours"]).any():
        raise ValueError("Duplicate condition/time observations found")
    if result[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing values found in required modelling columns")
    return result
