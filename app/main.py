from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.ml.model_loader import load_models
from app.ml.thresholds import load_thresholds
from app.services.inference_service import InferenceService
from app.api.routes import router as api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()

    logger.info(f"Environment: {settings.env}")
    logger.info(f"Using device: {settings.device_torch}")

    models = load_models()
    thresholds = load_thresholds()

    concept_order = [
        "rule_free_space",
        "rule_cable_routing",
        "rule_alignment",
        "rule_covering",
    ]

    app.state.models = models
    app.state.thresholds = thresholds
    app.state.inference_service = InferenceService(
        models=models,
        thresholds=thresholds,
        concept_order=concept_order,
    )

    logger.info("Inference service ready")
    yield

    # Shutdown (optioneel)
    # Als je later GPU resources expliciet wil vrijgeven/caches legen, kan dat hier.
    # Bijvoorbeeld: torch.cuda.empty_cache() (alleen als cuda)


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()
