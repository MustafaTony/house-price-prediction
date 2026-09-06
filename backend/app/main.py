"""
main.py

FastAPI application entrypoint.

The trained model is loaded exactly once, during application startup,
via the `lifespan` context manager below -- NOT on every request.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import prediction as prediction_routes
from app.core.config import settings
from app.services.inference import model_service
from app.utils.logging_config import configure_logging

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up %s v%s", settings.APP_NAME, settings.APP_VERSION)
    model_service.load()  # <-- loaded once here, reused for every request
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_routes.router, tags=["prediction"])
