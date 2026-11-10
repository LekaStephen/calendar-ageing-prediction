from pathlib import Path
import pandas as pd

class DatasetError(RuntimeError):
    pass

def load_processed_dataset(path):
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Processed dataset not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise DatasetError("Processed dataset is empty.")
    return frame
