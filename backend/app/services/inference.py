"""
services/inference.py

Loads the trained pipeline (house_price.pkl) ONCE and exposes prediction
functionality. The actual loading is triggered from main.py's lifespan
handler at application startup, not on a per-request basis.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Set

import joblib
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """
    Thin wrapper around the trained sklearn Pipeline. A single instance is
    created and its `load()` is called once during FastAPI startup
    (lifespan), then reused for every request -- the pipeline is never
    reloaded per-request.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._known_locations: Set[str] = set()

    def load(self) -> None:
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Run the ML training pipeline (src/train_model.py) first "
                "and ensure house_price.pkl is copied into backend/models/."
            )
        logger.info("Loading model from %s", model_path)
        self._pipeline = joblib.load(model_path)

        locations_path = Path(settings.LOCATIONS_PATH)
        if locations_path.exists():
            with open(locations_path) as f:
                self._known_locations = {loc.strip().lower() for loc in json.load(f)}
            logger.info("Loaded %d known real locations", len(self._known_locations))
        else:
            logger.warning("locations.json not found at %s -- "
                            "location_recognized will always report False", locations_path)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def known_locations(self) -> Set[str]:
        return self._known_locations

    def predict(self, feature_row: pd.DataFrame) -> float:
        if self._pipeline is None:
            raise RuntimeError("Model has not been loaded yet. Call load() at startup.")
        prediction = self._pipeline.predict(feature_row)
        return float(prediction[0])


# Module-level singleton, populated by main.py's lifespan handler.
model_service = ModelService()
