"""
services/preprocessing.py

Builds the one-row pandas DataFrame with exactly the feature names/dtypes
the trained Pipeline expects, from a validated PredictionRequest.

LOCATION GROUPING (project guide requirement -- applied here, not just
documented): the model was trained on the Top-50 most frequent real
locations (by training-split frequency) plus a literal "other" category
for everything else -- see src/train_model.py's use of
compute_top_n_locations()/apply_location_grouping() during training.
`models/locations.json` (copied to `backend/models/locations.json`)
contains exactly that Top-50 list, so `model_service.known_locations`
IS the same train-derived set used to decide the grouping during
training. `map_location_to_group()` below applies the identical rule at
inference time: any location not in that set -- including a location
that's real but outside the Top 50, and any location the model has
literally never seen -- is mapped to "other" before the feature row is
built, exactly as training did. This makes "other" a real, meaningful
model category rather than a zeroed-out unknown.

Every OTHER default used here for fields NOT exposed on the public form
is chosen to reproduce -- not invent -- how the training pipeline
actually handled a real listing that didn't report that information:

  - car_parking_count / car_parking_type -> left as missing (NaN / None).
    In training, ~55% of real listings had no Car Parking value at all;
    the fitted SimpleImputer inside the model's ColumnTransformer already
    learned how to fill those (median count, most-frequent type) from
    real data. Passing NaN here re-uses that exact learned behavior
    instead of us guessing a number in application code.

  - total_floors -> left as missing (NaN), for the same reason: many real
    "Floor" values were just a bare number with no "out of N" part, so
    total_floors was NaN for those real rows too, and the trained median
    imputer already handles it.

  - overlooks_garden_park / overlooks_pool / overlooks_main_road -> 0.
    In training, these flags come from clean_real_dataset(), which sets
    them to 0 whenever the raw `overlooking` field was missing or "Not
    Available" (81,436 / 187,531 raw rows). Defaulting to 0 here is the
    same transformation applied to any real listing that didn't specify
    this, not a new assumption.

  - has_society -> 0. Same logic: clean_real_dataset() sets this to 0
    whenever the raw `Society` field was missing (109,678 / 187,531 raw
    rows -- the majority case).

  - area_is_carpet_area -> 1, area_missing -> 0. The API explicitly asks
    for *carpet* area (matching the `Carpet Area` raw column), so this is
    not a default at all -- it's the same value clean_real_dataset()
    produces for any real row where carpet area was actually reported.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from app.schemas.prediction import PredictionRequest

OTHER_LOCATION_LABEL = "other"

# Exact column order the trained pipeline was fit on
# (see ColumnTransformer.feature_names_in_ inspected from house_price.pkl).
MODEL_FEATURE_COLUMNS = [
    "location", "Transaction", "Furnishing", "facing", "Ownership",
    "area_sqft", "area_is_carpet_area", "area_missing",
    "bathroom_num", "balcony_num",
    "car_parking_count", "car_parking_type",
    "floor_num", "total_floors",
    "overlooks_garden_park", "overlooks_pool", "overlooks_main_road",
    "has_society",
]


def map_location_to_group(location: str, known_locations: set[str]) -> str:
    """
    Applies the SAME Top-50-locations-else-'other' grouping used during
    training. `known_locations` must be the Top-50 set loaded from
    locations.json (see services/inference.py), which is itself derived
    from the training split only -- never recomputed here, just applied.
    """
    normalized = location.strip().lower()
    return normalized if normalized in known_locations else OTHER_LOCATION_LABEL


def build_feature_row(request: PredictionRequest, known_locations: set[str]) -> pd.DataFrame:
    """
    Converts a validated PredictionRequest into the one-row DataFrame the
    trained Pipeline expects, with columns in the exact names/order it
    was fit on. The location is grouped into the Top-50-or-"other"
    category the model was actually trained on before being placed in
    the row.
    """
    grouped_location = map_location_to_group(request.location, known_locations)

    row: Dict[str, Any] = {
        "location": grouped_location,
        "Transaction": request.transaction,
        "Furnishing": request.furnishing,
        "facing": request.facing,
        "Ownership": request.ownership,
        "area_sqft": float(request.carpet_area_sqft),
        "area_is_carpet_area": 1,
        "area_missing": 0,
        "bathroom_num": float(request.bathrooms),
        "balcony_num": float(request.balconies),
        "car_parking_count": np.nan,
        "car_parking_type": None,
        "floor_num": float(request.floor),
        "total_floors": np.nan,
        "overlooks_garden_park": 0,
        "overlooks_pool": 0,
        "overlooks_main_road": 0,
        "has_society": 0,
    }

    df = pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)
    return df


def is_known_location(location: str, known_locations: set[str]) -> bool:
    """
    Reports whether the given location is one of the Top-50 real
    locations the model was trained on as its own category (as opposed
    to being grouped into "other"). This is informational for the API
    response (`location_recognized`) -- the actual grouping behavior is
    applied unconditionally by build_feature_row()/map_location_to_group()
    regardless of this flag.
    """
    return location.strip().lower() in known_locations
