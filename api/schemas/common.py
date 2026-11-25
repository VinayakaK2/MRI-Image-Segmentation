"""Shared Pydantic schemas for API errors and health check responses."""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    """Structured error payload without leaking internal filesystem paths."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: Optional[Dict[str, object]] = None


class ErrorResponse(BaseModel):
    """Standard error envelope returned on validation, resource, or runtime failures."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


class ArtifactStatus(BaseModel):
    """Availability and validity status for a single research artifact."""

    model_config = ConfigDict(extra="forbid")

    name: str
    available: bool
    valid: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response reporting runtime, PyTorch, model, and artifact readiness."""

    model_config = ConfigDict(extra="forbid")

    status: str
    backend_version: str
    python_version: str
    pytorch_available: bool
    pytorch_version: Optional[str] = None
    model_initialized: bool
    canonical_checkpoint_id: str
    loaded_checkpoint_ids: List[str]
    model_load_error: Optional[str] = None
    device_active: str
    cuda_available: bool
    cuda_device_count: int
    research_artifacts_available: bool
    artifacts: List[ArtifactStatus]
