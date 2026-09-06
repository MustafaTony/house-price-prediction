"""
tests/test_prediction.py

Uses FastAPI's TestClient (httpx-based) as a context manager so the
app's lifespan actually runs (loading the real trained model once,
exactly as it would in production) -- constructing TestClient(app)
without the `with` block does NOT trigger lifespan startup/shutdown.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_request_returns_success(client):
    payload = {
        "location": "gurgaon",
        "carpet_area_sqft": 1200,
        "floor": 3,
        "bathrooms": 2,
        "balconies": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_price_rupees" in body
    assert isinstance(body["predicted_price_rupees"], (int, float))
    assert body["predicted_price_rupees"] > 0
    assert body["location_recognized"] is True  # 'gurgaon' is a real trained location


def test_predict_unknown_location_still_succeeds_safely(client):
    """Unseen locations must not error -- grouped into 'other', same as training."""
    payload = {
        "location": "some-city-not-in-training-data",
        "carpet_area_sqft": 900,
        "floor": 1,
        "bathrooms": 1,
        "balconies": 1,
        "furnishing": "Unfurnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "North",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_price_rupees"] > 0
    assert body["location_recognized"] is False


def test_predict_real_but_non_top50_location_groups_into_other(client):
    """
    A REAL location from the original 81 that didn't make the Top-50 cut
    (e.g. 'madurai', one of the smallest-support real locations) must
    still succeed and be reported as not individually recognized -- it's
    grouped into 'other', exactly like a totally unseen location, not
    given special treatment for being 'real but rare'.
    """
    payload = {
        "location": "madurai",
        "carpet_area_sqft": 950,
        "floor": 2,
        "bathrooms": 2,
        "balconies": 1,
        "furnishing": "Unfurnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "South",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_price_rupees"] > 0
    assert body["location_recognized"] is False


def test_predict_invalid_request_missing_field_returns_422(client):
    payload = {
        # "location" intentionally omitted
        "carpet_area_sqft": 1200,
        "floor": 3,
        "bathrooms": 2,
        "balconies": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_area_returns_422(client):
    """area <= 0 must fail validation."""
    payload = {
        "location": "gurgaon",
        "carpet_area_sqft": 0,
        "floor": 3,
        "bathrooms": 2,
        "balconies": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_enum_value_returns_422(client):
    """furnishing must be one of the real trained categories."""
    payload = {
        "location": "gurgaon",
        "carpet_area_sqft": 1200,
        "floor": 3,
        "bathrooms": 2,
        "balconies": 1,
        "furnishing": "Fully-Loaded",  # not a real category
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422

