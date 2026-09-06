"""
api/routes/prediction.py

GET  /health  -- liveness/readiness check, reports whether the model is loaded.
POST /predict -- runs a real-time prediction through the trained pipeline.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.prediction import HealthResponse, PredictionRequest, PredictionResponse
from app.services.inference import model_service
from app.services.preprocessing import build_feature_row, is_known_location

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not model_service.is_loaded:
        # Should not happen in normal operation (model loads at startup),
        # but fail loudly and clearly rather than crash if it does.
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please retry shortly.",
        )

    known = is_known_location(request.location, model_service.known_locations)

    try:
        feature_row = build_feature_row(request, model_service.known_locations)
        predicted_price = model_service.predict(feature_row)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        predicted_price_rupees=predicted_price,
        predicted_price_formatted=f"\u20b9{predicted_price:,.0f}",
        location_recognized=known,
    )
