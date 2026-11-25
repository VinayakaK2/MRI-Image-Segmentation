"""Overview and model specification route for GET /api/v1/overview."""

from fastapi import APIRouter, Request

from api.schemas.overview import OverviewResponse

router = APIRouter(tags=["overview"])


@router.get("/overview", response_model=OverviewResponse)
def get_overview(request: Request) -> OverviewResponse:
    """Returns project overview, LightweightUNet2D specs, checkpoints, dataset counts, and baseline metrics."""
    return request.app.state.model_service.get_overview()
