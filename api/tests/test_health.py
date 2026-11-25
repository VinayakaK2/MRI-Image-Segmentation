"""Tests for GET /api/v1/health, endpoint surface, and CORS restrictions."""

from pathlib import Path
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import CANONICAL_CHECKPOINT_ID, build_research_paths


def test_health_endpoint_ready(client: TestClient) -> None:
    """Verifies backend starts and reports model, PyTorch, device, and artifact readiness."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["pytorch_available"] is True
    assert data["pytorch_version"] is not None
    assert data["model_initialized"] is True
    assert data["canonical_checkpoint_id"] == CANONICAL_CHECKPOINT_ID
    assert CANONICAL_CHECKPOINT_ID in data["loaded_checkpoint_ids"]
    assert data["model_load_error"] is None
    assert data["device_active"] in ("cpu", "cuda")
    assert data["research_artifacts_available"] is True
    assert len(data["artifacts"]) >= 5
    assert all(a["available"] and a["valid"] for a in data["artifacts"])


def test_health_reports_not_ready_when_checkpoint_missing(tmp_path: Path) -> None:
    """Verifies health never claims model readiness if checkpoint loading fails."""
    # Point project_root to an empty temp directory with missing checkpoint/artifacts
    isolated_paths = build_research_paths(project_root=tmp_path)
    app = create_app(paths=isolated_paths)
    with TestClient(app) as isolated_client:
        resp = isolated_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_initialized"] is False
        assert data["status"] in ("degraded", "unavailable")
        assert data["model_load_error"] is not None
        assert data["loaded_checkpoint_ids"] == []
        assert data["research_artifacts_available"] is False


def test_exactly_five_api_endpoints_registered(client: TestClient) -> None:
    """Verifies only the five specified /api/v1 endpoints are exposed in OpenAPI."""
    openapi = client.get("/openapi.json").json()
    paths = openapi["paths"]
    api_routes = {
        (path, tuple(sorted(m.upper() for m in methods.keys())))
        for path, methods in paths.items()
    }
    expected = {
        ("/api/v1/health", ("GET",)),
        ("/api/v1/overview", ("GET",)),
        ("/api/v1/patients", ("GET",)),
        ("/api/v1/inference/segment", ("POST",)),
        ("/api/v1/experiments", ("GET",)),
    }
    assert api_routes == expected


def test_cors_allows_local_dev_and_blocks_arbitrary_origin(client: TestClient) -> None:
    """Verifies CORS permits localhost frontend origin and does not allow arbitrary public origins."""
    ok_resp = client.options(
        "/api/v1/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert ok_resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    blocked_resp = client.options(
        "/api/v1/health",
        headers={"Origin": "https://untrusted.example.com", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in blocked_resp.headers
