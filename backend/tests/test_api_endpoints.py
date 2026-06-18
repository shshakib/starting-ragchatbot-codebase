"""Tests for the FastAPI API endpoints (/api/query, /api/courses, /).

The real app in backend/app.py mounts static files from "../frontend" on import,
which don't exist in the test environment. These tests instead exercise a
test-only FastAPI app (see the `test_app` / `client` fixtures in conftest.py)
that mirrors the same endpoint definitions but wires in a mocked RAGSystem.
"""
import pytest


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

def test_query_returns_200(client):
    response = client.post("/api/query", json={"query": "What is AI?"})

    assert response.status_code == 200


def test_query_response_contains_expected_fields(client):
    response = client.post("/api/query", json={"query": "What is AI?"})

    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "session_id" in data


def test_query_returns_answer_from_rag_system(client, mock_rag_system):
    mock_rag_system.query.return_value = ("42 is the answer.", [])

    response = client.post("/api/query", json={"query": "What is the answer?"})

    assert response.json()["answer"] == "42 is the answer."


def test_query_returns_sources_from_rag_system(client, mock_rag_system):
    mock_rag_system.query.return_value = (
        "Some answer.",
        [{"text": "Python 101 - Lesson 1", "url": "https://example.com"}],
    )

    response = client.post("/api/query", json={"query": "test"})

    assert response.json()["sources"] == [
        {"text": "Python 101 - Lesson 1", "url": "https://example.com"}
    ]


def test_query_without_session_id_creates_new_session(client, mock_rag_system):
    mock_rag_system.session_manager.create_session.return_value = "new-session-42"

    response = client.post("/api/query", json={"query": "test"})

    mock_rag_system.session_manager.create_session.assert_called_once()
    assert response.json()["session_id"] == "new-session-42"


def test_query_with_session_id_reuses_it(client, mock_rag_system):
    response = client.post(
        "/api/query", json={"query": "test", "session_id": "existing-session"}
    )

    mock_rag_system.session_manager.create_session.assert_not_called()
    assert response.json()["session_id"] == "existing-session"


def test_query_passes_query_and_session_to_rag_system(client, mock_rag_system):
    client.post("/api/query", json={"query": "Explain backprop", "session_id": "s1"})

    mock_rag_system.query.assert_called_once_with("Explain backprop", "s1")


def test_query_missing_query_field_returns_422(client):
    response = client.post("/api/query", json={"session_id": "s1"})

    assert response.status_code == 422


def test_query_rag_system_exception_returns_500(client, mock_rag_system):
    mock_rag_system.query.side_effect = Exception("vector store unavailable")

    response = client.post("/api/query", json={"query": "test"})

    assert response.status_code == 500
    assert "vector store unavailable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

def test_get_courses_returns_200(client):
    response = client.get("/api/courses")

    assert response.status_code == 200


def test_get_courses_returns_analytics_fields(client, mock_rag_system):
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["A", "B", "C"],
    }

    response = client.get("/api/courses")

    data = response.json()
    assert data["total_courses"] == 3
    assert data["course_titles"] == ["A", "B", "C"]


def test_get_courses_exception_returns_500(client, mock_rag_system):
    mock_rag_system.get_course_analytics.side_effect = Exception("db error")

    response = client.get("/api/courses")

    assert response.status_code == 500
    assert "db error" in response.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

def test_delete_session_returns_200(client):
    response = client.delete("/api/session/some-session-id")

    assert response.status_code == 200
    assert response.json() == {"status": "cleared"}


def test_delete_session_calls_clear_session(client, mock_rag_system):
    client.delete("/api/session/abc-123")

    mock_rag_system.session_manager.clear_session.assert_called_once_with("abc-123")


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

def test_root_returns_200(client):
    response = client.get("/")

    assert response.status_code == 200
