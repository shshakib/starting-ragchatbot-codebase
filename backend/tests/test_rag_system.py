"""Tests for RAGSystem.query() content-query handling in rag_system.py"""

import pytest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_mock_config():
    cfg = MagicMock()
    cfg.ANTHROPIC_API_KEY = "fake-key"
    cfg.ANTHROPIC_MODEL = "claude-test-model"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.CHROMA_PATH = "./chroma_db"
    return cfg


@pytest.fixture()
def rag(tmp_path):
    """
    Return a RAGSystem with ALL heavy dependencies replaced by mocks:
    - VectorStore  (no ChromaDB, no sentence-transformers)
    - AIGenerator  (no Anthropic API calls)
    - SessionManager
    - ToolManager + tools (registered automatically inside RAGSystem.__init__)
    """
    with (
        patch("rag_system.VectorStore") as MockVS,
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.SessionManager") as MockSM,
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.ToolManager") as MockTM,
        patch("rag_system.CourseSearchTool"),
        patch("rag_system.CourseOutlineTool"),
    ):
        from rag_system import RAGSystem

        cfg = make_mock_config()
        system = RAGSystem(cfg)

        # Expose mock instances for test assertions
        system._mock_ai = MockAI.return_value
        system._mock_sm = MockSM.return_value
        system._mock_tm = MockTM.return_value

        # Default happy-path behaviour
        system._mock_ai.generate_response.return_value = "Here is your answer."
        system._mock_tm.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        system._mock_tm.get_last_sources.return_value = []

        yield system


# ---------------------------------------------------------------------------
# query() — basic contract
# ---------------------------------------------------------------------------


def test_query_returns_tuple_of_response_and_sources(rag):
    result = rag.query("What is machine learning?")

    assert isinstance(result, tuple)
    assert len(result) == 2
    response, sources = result
    assert isinstance(response, str)
    assert isinstance(sources, list)


def test_query_response_comes_from_ai_generator(rag):
    rag._mock_ai.generate_response.return_value = "AI generated answer."

    response, _ = rag.query("test question")

    assert response == "AI generated answer."


def test_query_sources_come_from_tool_manager(rag):
    rag._mock_tm.get_last_sources.return_value = [
        {"text": "Python 101 - Lesson 1", "url": "https://example.com"}
    ]

    _, sources = rag.query("test question")

    assert sources == [{"text": "Python 101 - Lesson 1", "url": "https://example.com"}]


# ---------------------------------------------------------------------------
# query() — tool wiring
# ---------------------------------------------------------------------------


def test_query_passes_tool_definitions_to_ai_generator(rag):
    tool_defs = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
    rag._mock_tm.get_tool_definitions.return_value = tool_defs

    rag.query("What is AI?")

    call_kwargs = rag._mock_ai.generate_response.call_args[1]
    assert call_kwargs["tools"] == tool_defs


def test_query_passes_tool_manager_to_ai_generator(rag):
    rag.query("What is AI?")

    call_kwargs = rag._mock_ai.generate_response.call_args[1]
    assert call_kwargs["tool_manager"] is rag._mock_tm


def test_query_resets_sources_after_retrieval(rag):
    rag.query("Some question")

    rag._mock_tm.reset_sources.assert_called_once()


def test_query_retrieves_sources_before_reset(rag):
    """get_last_sources must be called before reset_sources."""
    call_order = []
    rag._mock_tm.get_last_sources.side_effect = lambda: call_order.append("get")
    rag._mock_tm.reset_sources.side_effect = lambda: call_order.append("reset")

    rag.query("test")

    assert call_order.index("get") < call_order.index("reset")


# ---------------------------------------------------------------------------
# query() — session / history handling
# ---------------------------------------------------------------------------


def test_query_saves_exchange_when_session_id_provided(rag):
    rag._mock_ai.generate_response.return_value = "The answer."

    rag.query("My question", session_id="sess-1")

    rag._mock_sm.add_exchange.assert_called_once_with("sess-1", "My question", "The answer.")


def test_query_no_session_does_not_save_exchange(rag):
    rag.query("My question", session_id=None)

    rag._mock_sm.add_exchange.assert_not_called()


def test_query_with_session_fetches_history(rag):
    rag._mock_sm.get_conversation_history.return_value = "User: hi\nAI: hello"

    rag.query("follow-up", session_id="sess-2")

    rag._mock_sm.get_conversation_history.assert_called_once_with("sess-2")
    call_kwargs = rag._mock_ai.generate_response.call_args[1]
    assert call_kwargs["conversation_history"] == "User: hi\nAI: hello"


def test_query_no_session_passes_no_history_to_ai(rag):
    rag.query("question", session_id=None)

    call_kwargs = rag._mock_ai.generate_response.call_args[1]
    assert call_kwargs.get("conversation_history") is None


# ---------------------------------------------------------------------------
# query() — content-query end-to-end mock flow
# ---------------------------------------------------------------------------


def test_query_content_question_full_tool_flow(rag):
    """
    Simulate the full RAG flow:
    AI is called → decides to use tool → tool executes → final answer returned.
    In this test we verify that RAGSystem wires all pieces together correctly
    without making any real API or DB calls.
    """
    # AIGenerator simulates a full tool-use cycle and returns a synthesized answer
    rag._mock_ai.generate_response.return_value = (
        "Backpropagation is an algorithm for computing gradients."
    )
    rag._mock_tm.get_last_sources.return_value = [
        {"text": "Deep Learning - Lesson 3", "url": "https://example.com/lesson/3"}
    ]

    response, sources = rag.query("Explain backpropagation", session_id="test-session")

    # Correct answer produced
    assert "Backpropagation" in response
    # Sources correctly surfaced
    assert sources[0]["text"] == "Deep Learning - Lesson 3"
    # Session history recorded
    rag._mock_sm.add_exchange.assert_called_once()
    # Sources reset after the query
    rag._mock_tm.reset_sources.assert_called_once()


def test_query_prompt_wraps_user_question(rag):
    """RAGSystem wraps the raw query in a prompt before passing to AIGenerator."""
    rag.query("What is a transformer?")

    call_kwargs = rag._mock_ai.generate_response.call_args[1]
    query_sent = call_kwargs["query"]
    assert "What is a transformer?" in query_sent


def test_query_empty_sources_returned_when_no_tool_used(rag):
    rag._mock_tm.get_last_sources.return_value = []

    _, sources = rag.query("What is 2 + 2?")

    assert sources == []
