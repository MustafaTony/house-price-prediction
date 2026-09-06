"""
predict.py

Loads the trained model (models/house_price.pkl) and, if present,
outputs/locations.json, to make predictions on new real input rows.

This module does nothing until train_model.py has been successfully run
on the real dataset -- it will raise a clear error otherwise.
"""

import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "house_price.pkl"
LOCATIONS_PATH = ROOT / "outputs" / "locations.json"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run src/train_model.py on the real dataset first."
        )
    return joblib.load(MODEL_PATH)


def load_locations():
    if not LOCATIONS_PATH.exists():
        return None
    with open(LOCATIONS_PATH) as f:
        return json.load(f)


def predict(input_rows: list[dict]) -> list[float]:
    """
    input_rows: list of dicts matching the real feature schema the model
    was trained on (same column names, minus the target column).
    """
    model = load_model()
    df = pd.DataFrame(input_rows)
    return model.predict(df).tolist()


if __name__ == "__main__":
    locations = load_locations()
    print("Available locations:" if locations else "No locations.json found yet.")
    if locations:
        print(locations[:20], "..." if len(locations) > 20 else "")
