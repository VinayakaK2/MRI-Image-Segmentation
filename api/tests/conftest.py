"""Shared pytest fixtures for local FastAPI endpoint integration tests."""

from typing import Iterator
from fastapi.testclient import TestClient
import pytest

from api.app import create_app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Provides a module-scoped TestClient so the canonical model loads once per test module."""
    application = create_app()
    with TestClient(application) as test_client:
        yield test_client
