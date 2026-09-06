"""
core/config.py

Application configuration loaded from environment variables / .env via
pydantic-settings. Nothing here is hardcoded that should differ between
environments (model path, allowed CORS origins, etc.).
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "House Price Prediction API"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    # Path to the trained pipeline (RandomForest winner from the ML phase).
    MODEL_PATH: str = str(BACKEND_ROOT / "models" / "house_price.pkl")

    # Path to the real locations exported during training (used only for
    # reference/validation context, not required for the model to run --
    # the model itself handles unseen locations via handle_unknown="ignore").
    LOCATIONS_PATH: str = str(BACKEND_ROOT / "models" / "locations.json")

    # Comma-separated list of allowed CORS origins.
    CORS_ORIGINS: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
