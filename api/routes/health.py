"""Health check route for GET /api/v1/health."""

from fastapi import APIRouter, Request

from api.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """Returns backend, PyTorch, model initialization, device, and artifact readiness."""
    return request.app.state.model_service.get_health_status()
