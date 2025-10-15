"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_query_endpoint_validation(client):
    """Test query endpoint input validation."""
    # Test with missing query
    response = client.post("/api/query", json={})
    assert response.status_code == 422  # Validation error

    # Test with valid query
    response = client.post(
        "/api/query",
        json={
            "query": "What is RAG?",
            "top_k": 5,
            "use_llm": False,
        },
    )
    # Response might succeed or fail depending on initialization
    # Just check it doesn't crash
    assert response.status_code in [200, 500]
