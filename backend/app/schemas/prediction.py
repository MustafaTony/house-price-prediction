"""
schemas/prediction.py

Request/response schemas for POST /predict.

IMPORTANT -- schema design note:
The trained pipeline (see models/house_price.pkl) actually expects 18
internal feature columns:
    location, Transaction, Furnishing, facing, Ownership,
    area_sqft, area_is_carpet_area, area_missing,
    bathroom_num, balcony_num,
    car_parking_count, car_parking_type,
    floor_num, total_floors,
    overlooks_garden_park, overlooks_pool, overlooks_main_road,
    has_society

Per the project guide, the public-facing request only collects the
fields a real user can reasonably provide on a listing form: location,
carpet area, floor, bathrooms, balconies, furnishing, transaction,
ownership, facing. The remaining internal columns (car parking,
total floors, overlooking flags, society flag, area-source flags) are
NOT user inputs -- they are derived/defaulted in
services/preprocessing.py using the exact same logic the training
pipeline used for real listings that didn't report that information
(see that module's docstring for the full justification). This keeps
the API contract honest about what it actually asks the user for, while
still building a row that matches the model's real training schema.

The categorical Literal values below are the exact real category values
observed in the real Kaggle dataset after cleaning (see
src/feature_engineering.py / clean_real_dataset) -- not invented values.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Furnishing = Literal["Furnished", "Semi-Furnished", "Unfurnished"]
Transaction = Literal["New Property", "Resale", "Rent/Lease", "Other"]
Ownership = Literal["Freehold", "Leasehold", "Co-operative Society", "Power Of Attorney"]
Facing = Literal[
    "East", "North", "North - East", "North - West",
    "South", "South - East", "South -West", "West",
]


class PredictionRequest(BaseModel):
    location: str = Field(
        ..., min_length=1, max_length=100,
        description="City/area name. The model was trained on the Top-50 "
                    "most frequent real locations plus an 'other' category; "
                    "any location outside that Top-50 (including ones never "
                    "seen at all) is safely grouped into 'other' -- the same "
                    "grouping used during training, not just a fallback.",
        examples=["gurgaon"],
    )
    carpet_area_sqft: float = Field(
        ..., gt=0, le=1_000_000,
        description="Carpet area in square feet. Must be greater than 0.",
        examples=[1200.0],
    )
    floor: int = Field(
        ..., ge=-2, le=250,
        description="The floor the unit is on (0 = ground floor, "
                    "-1 = lower basement).",
        examples=[3],
    )
    bathrooms: int = Field(..., ge=0, le=20, examples=[2])
    balconies: int = Field(..., ge=0, le=20, examples=[1])
    furnishing: Furnishing = Field(..., examples=["Semi-Furnished"])
    transaction: Transaction = Field(..., examples=["Resale"])
    ownership: Ownership = Field(..., examples=["Freehold"])
    facing: Facing = Field(..., examples=["East"])

    @field_validator("location")
    @classmethod
    def normalize_location(cls, v: str) -> str:
        return v.strip().lower()


class PredictionResponse(BaseModel):
    predicted_price_rupees: float
    predicted_price_formatted: str
    currency: str = "INR"
    model_used: str = "RandomForestRegressor"
    location_recognized: bool = Field(
        ...,
        description="False if the given location was not one of the "
                    "locations seen during training (prediction still "
                    "succeeds -- unseen locations are handled safely).",
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    app_name: str
    version: str
