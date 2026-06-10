import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient that does NOT trigger the lifespan (no Neo4j needed).

    Individual tests monkeypatch app.graph.* and app.tasks.* before issuing
    requests so no live service connections are made.
    """
    from app.main import app

    # Not using the context-manager form intentionally — lifespan (constraint
    # setup, index creation) must not run during unit tests (DECISIONS.md).
    return TestClient(app)
