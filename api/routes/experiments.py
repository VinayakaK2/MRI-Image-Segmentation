"""Phase 4 robustness and Phase 5 mitigation experiments route for GET /api/v1/experiments."""

from typing import Optional
from fastapi import APIRouter, Query, Request

from api.schemas.experiment import ExperimentsResponse

router = APIRouter(tags=["experiments"])


@router.get("/experiments", response_model=ExperimentsResponse)
def get_experiments(
    request: Request,
    phase: Optional[int] = Query(
        default=None,
        description="Optional experiment phase filter: 4 (robustness) or 5 (mitigation).",
    ),
    family: Optional[str] = Query(
        default=None,
        description="Optional corruption family (Phase 4) or registered hypothesis family (Phase 5).",
    ),
    patient_id: Optional[str] = Query(
        default=None,
        description="Optional test-cohort patient ID filter for patient-level records.",
    ),
) -> ExperimentsResponse:
    """Returns Phase 4 robustness and Phase 5 mitigation results directly from research JSON/CSV artifacts."""
    return request.app.state.experiment_service.get_experiments(
        phase=phase,
        family=family,
        patient_id=patient_id,
    )
