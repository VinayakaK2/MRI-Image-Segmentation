"""Synchronous segmentation and stress-test route for POST /api/v1/inference/segment."""

from fastapi import APIRouter, Request

from api.schemas.inference import SegmentRequest, SegmentResponse

router = APIRouter(tags=["inference"])


@router.post("/inference/segment", response_model=SegmentResponse)
def post_inference_segment(request: Request, payload: SegmentRequest) -> SegmentResponse:
    """Executes synchronous single-slice cardiac MRI segmentation with optional corruption and mitigation."""
    return request.app.state.inference_service.run_segmentation(payload)
