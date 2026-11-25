"""FastAPI application factory, lifespan model initialization, CORS, and safe error handlers."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator, Optional
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import (
    API_PREFIX,
    BACKEND_VERSION,
    PROJECT_TITLE,
    ResearchPaths,
    build_research_paths,
    get_allowed_cors_origins,
)
from api.routes import (
    experiments_router,
    health_router,
    inference_router,
    overview_router,
    patients_router,
)
from api.schemas.common import ErrorDetail, ErrorResponse
from api.services import ExperimentService, InferenceService, ModelService, PatientService
from api.utils.validation import ApiError

logger = logging.getLogger("cardiac_mri_api")


def create_app(paths: Optional[ResearchPaths] = None) -> FastAPI:
    """Builds and configures the local FastAPI application with exactly five API endpoints."""
    resolved_paths = paths or build_research_paths()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        """Initializes shared services and loads the canonical checkpoint once at startup."""
        model_service = ModelService(paths=resolved_paths)
        patient_service = PatientService(paths=resolved_paths)
        inference_service = InferenceService(
            model_service=model_service,
            patient_service=patient_service,
        )
        experiment_service = ExperimentService(paths=resolved_paths)

        model_service.initialize_canonical_model()

        app_instance.state.model_service = model_service
        app_instance.state.patient_service = patient_service
        app_instance.state.inference_service = inference_service
        app_instance.state.experiment_service = experiment_service
        yield

    application = FastAPI(
        title=PROJECT_TITLE,
        version=BACKEND_VERSION,
        lifespan=lifespan,
    )

    # Restrict CORS strictly to discovered/configured local frontend dev origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @application.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        """Serializes structured domain errors without exposing internal filesystem paths."""
        body = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """Formats Pydantic request validation errors into the unified ErrorResponse schema."""
        sanitized_errors = [
            {
                "loc": list(err.get("loc", ())),
                "msg": str(err.get("msg", "Invalid input")),
                "type": str(err.get("type", "value_error")),
            }
            for err in exc.errors()
        ]
        body = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request validation failed.",
                details={"errors": sanitized_errors},
            )
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        """Logs unexpected exceptions internally and returns a safe generic 500 JSON payload."""
        logger.exception("Unhandled backend exception during request processing: %s", type(exc).__name__)
        body = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal error occurred while processing the request.",
            )
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    # Register strictly the five required API endpoints under /api/v1
    application.include_router(health_router, prefix=API_PREFIX)
    application.include_router(overview_router, prefix=API_PREFIX)
    application.include_router(patients_router, prefix=API_PREFIX)
    application.include_router(inference_router, prefix=API_PREFIX)
    application.include_router(experiments_router, prefix=API_PREFIX)

    return application


app = create_app()
