"""Held-out test patients and ED slices route for GET /api/v1/patients."""

from typing import Optional
from fastapi import APIRouter, Query, Request

from api.schemas.patient import PatientsResponse

router = APIRouter(tags=["patients"])


@router.get("/patients", response_model=PatientsResponse)
def get_patients(
    request: Request,
    patient_id: Optional[str] = Query(
        default=None,
        description="Optional test-cohort patient identifier (e.g. 'patient001').",
    ),
) -> PatientsResponse:
    """Returns the 15-patient held-out test cohort and available ED slices from acdc_slice_manifest.json."""
    return request.app.state.patient_service.list_test_patients(patient_id=patient_id)
