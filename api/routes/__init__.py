"""Router exports for the five API endpoints."""

from api.routes.experiments import router as experiments_router
from api.routes.health import router as health_router
from api.routes.inference import router as inference_router
from api.routes.overview import router as overview_router
from api.routes.patients import router as patients_router

__all__ = [
    "experiments_router",
    "health_router",
    "inference_router",
    "overview_router",
    "patients_router",
]
