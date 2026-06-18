import sys
import os
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

# Make backend/ importable so that "from vector_store import ..." works in tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture()
def mock_rag_system():
    """A MagicMock standing in for RAGSystem, wired with sane default returns."""
    mock = MagicMock()
    mock.query.return_value = ("This is a test answer.", [])
    mock.session_manager.create_session.return_value = "test-session-id"
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python 101", "Deep Learning"],
    }
    return mock


@pytest.fixture()
def test_app(mock_rag_system):
    """
    Build a FastAPI app that mirrors backend/app.py's API endpoints, but without
    mounting static files (which don't exist in the test environment) and using
    an injected mock RAGSystem instead of constructing a real one.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Course Materials RAG System - Test")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[dict]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        mock_rag_system.session_manager.clear_session(session_id)
        return {"status": "cleared"}

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}

    return app


@pytest.fixture()
def client(test_app):
    """A TestClient bound to the test app."""
    from fastapi.testclient import TestClient

    return TestClient(test_app)
