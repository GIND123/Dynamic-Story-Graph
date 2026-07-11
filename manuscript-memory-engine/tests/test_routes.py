"""Smoke tests for FastAPI routes — all external dependencies are monkeypatched."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


_MANUSCRIPT_PAYLOAD = {
    "title": "The Ashen Crown",
    "premise": "A city-state after its king's death",
    "characters": [
        {"name": "Mara", "traits": "pragmatic smuggler", "location": "Duskwall docks"}
    ],
    "rules": ["The dead do not return"],
}


class TestHealthRoute:
    def test_health_all_ok(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.main._check_neo4j", lambda: "ok")
        monkeypatch.setattr("app.main._check_redis", lambda: "ok")

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["components"]["neo4j"] == "ok"
        assert data["components"]["redis"] == "ok"

    def test_health_degraded_when_neo4j_down(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.main._check_neo4j", lambda: "error")
        monkeypatch.setattr("app.main._check_redis", lambda: "ok")

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


class TestManuscriptRoutes:
    def test_create_manuscript_returns_id(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.graph.create_manuscript", lambda body: "testmid001")

        resp = client.post("/manuscripts", json=_MANUSCRIPT_PAYLOAD)
        assert resp.status_code == 200
        assert resp.json()["manuscript_id"] == "testmid001"

    def test_get_state_returns_dict(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "app.graph.get_state",
            lambda mid: {"characters": [], "events": [], "rules": []},
        )

        resp = client.get("/manuscripts/testmid001/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "characters" in data
        assert "events" in data
        assert "rules" in data


class TestChapterRoutes:
    def test_request_chapter_returns_202(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.graph.is_eval_manuscript", lambda mid: False)
        monkeypatch.setattr("app.graph.next_chapter_number", lambda mid: 1)

        mock_task = MagicMock()
        mock_task.id = "fake-job-uuid"
        monkeypatch.setattr(
            "app.tasks.generate_chapter_task",
            MagicMock(delay=MagicMock(return_value=mock_task)),
        )

        resp = client.post("/manuscripts/testmid001/chapters", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"] == "fake-job-uuid"
        assert data["chapter_number"] == 1

    def test_request_chapter_rejected_for_eval_manuscript(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Eval-only (source='eval') manuscripts must never be generated into."""
        monkeypatch.setattr("app.graph.is_eval_manuscript", lambda mid: True)

        resp = client.post("/manuscripts/pdnc:SomeNovel/chapters", json={})
        assert resp.status_code == 403
        assert "eval" in resp.json()["detail"].lower()

    def test_get_chapter_returns_chapter(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "app.graph.get_chapter",
            lambda mid, n: {
                "number": 1,
                "status": "COMMITTED",
                "text": "Once upon a time...",
            },
        )

        resp = client.get("/manuscripts/testmid001/chapters/1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "COMMITTED"

    def test_get_chapter_404_when_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.graph.get_chapter", lambda mid, n: None)

        resp = client.get("/manuscripts/testmid001/chapters/99")
        assert resp.status_code == 404


class TestJobRoute:
    def test_get_job_pending(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_result.ready.return_value = False
        mock_result.result = None

        monkeypatch.setattr(
            "app.main.AsyncResult",
            lambda job_id, app: mock_result,
        )

        resp = client.get("/jobs/fake-job-uuid")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "PENDING"
        assert data["result"] is None

    def test_get_job_success(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.ready.return_value = True
        mock_result.result = {"status": "pending_pipeline", "chapter_number": 1}

        monkeypatch.setattr(
            "app.main.AsyncResult",
            lambda job_id, app: mock_result,
        )

        resp = client.get("/jobs/fake-job-uuid")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert data["result"]["chapter_number"] == 1
