"""Tests for CourseSearchTool.execute() in search_tools.py"""

import pytest
from unittest.mock import MagicMock, patch
from vector_store import SearchResults
from search_tools import CourseSearchTool


def make_results(documents, metadata, error=None):
    """Helper to build SearchResults without touching ChromaDB."""
    return SearchResults(
        documents=documents,
        metadata=metadata,
        distances=[0.1] * len(documents),
        error=error,
    )


def make_tool():
    """Return a CourseSearchTool backed by a plain MagicMock VectorStore."""
    store = MagicMock()
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    return CourseSearchTool(store), store


# ---------------------------------------------------------------------------
# execute() output tests
# ---------------------------------------------------------------------------


def test_execute_returns_formatted_results():
    tool, store = make_tool()
    store.search.return_value = make_results(
        documents=["Content about Python loops."],
        metadata=[{"course_title": "Python 101", "lesson_number": 2}],
    )

    result = tool.execute(query="loops")

    assert "[Python 101 - Lesson 2]" in result
    assert "Content about Python loops." in result


def test_execute_empty_results_returns_no_content_message():
    tool, store = make_tool()
    store.search.return_value = make_results(documents=[], metadata=[])

    result = tool.execute(query="dragons")

    assert "No relevant content found" in result


def test_execute_empty_results_includes_course_filter_in_message():
    tool, store = make_tool()
    store.search.return_value = make_results(documents=[], metadata=[])

    result = tool.execute(query="dragons", course_name="MCP Course")

    assert "MCP Course" in result


def test_execute_empty_results_includes_lesson_filter_in_message():
    tool, store = make_tool()
    store.search.return_value = make_results(documents=[], metadata=[])

    result = tool.execute(query="intro", lesson_number=3)

    assert "lesson 3" in result.lower()


def test_execute_with_search_error_returns_error_string():
    tool, store = make_tool()
    store.search.return_value = make_results(
        documents=[], metadata=[], error="No course found matching 'XYZ'"
    )

    result = tool.execute(query="something", course_name="XYZ")

    assert "No course found matching 'XYZ'" in result


# ---------------------------------------------------------------------------
# Argument forwarding tests
# ---------------------------------------------------------------------------


def test_execute_passes_query_to_store_search():
    tool, store = make_tool()
    store.search.return_value = make_results(documents=[], metadata=[])

    tool.execute(query="neural networks")

    store.search.assert_called_once_with(
        query="neural networks", course_name=None, lesson_number=None
    )


def test_execute_passes_course_name_to_store_search():
    tool, store = make_tool()
    store.search.return_value = make_results(documents=[], metadata=[])

    tool.execute(query="layers", course_name="Deep Learning")

    _, kwargs = store.search.call_args
    assert kwargs["course_name"] == "Deep Learning"


def test_execute_passes_lesson_number_to_store_search():
    tool, store = make_tool()
    store.search.return_value = make_results(documents=[], metadata=[])

    tool.execute(query="backprop", lesson_number=5)

    _, kwargs = store.search.call_args
    assert kwargs["lesson_number"] == 5


# ---------------------------------------------------------------------------
# Source tracking tests
# ---------------------------------------------------------------------------


def test_execute_tracks_sources_after_successful_search():
    tool, store = make_tool()
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.search.return_value = make_results(
        documents=["Some content."],
        metadata=[{"course_title": "AI Basics", "lesson_number": 1}],
    )

    tool.execute(query="AI")

    assert len(tool.last_sources) == 1
    assert tool.last_sources[0]["text"] == "AI Basics - Lesson 1"
    assert tool.last_sources[0]["url"] == "https://example.com/lesson/1"


def test_execute_deduplicates_sources_for_same_course_lesson():
    tool, store = make_tool()
    store.search.return_value = make_results(
        documents=["Chunk A.", "Chunk B."],
        metadata=[
            {"course_title": "AI Basics", "lesson_number": 1},
            {"course_title": "AI Basics", "lesson_number": 1},
        ],
    )

    tool.execute(query="AI")

    assert len(tool.last_sources) == 1


def test_execute_sources_empty_when_no_results():
    tool, store = make_tool()
    store.search.return_value = make_results(documents=[], metadata=[])
    tool.last_sources = [{"text": "old", "url": None}]  # pre-existing stale value

    tool.execute(query="nothing")

    # last_sources should not be updated on empty result
    # (the code only sets last_sources inside _format_results, which is not called)
    assert tool.last_sources == [{"text": "old", "url": None}]


def test_execute_multiple_distinct_lessons_tracked_as_separate_sources():
    tool, store = make_tool()
    store.get_lesson_link.side_effect = [
        "https://example.com/lesson/1",
        "https://example.com/lesson/2",
    ]
    store.search.return_value = make_results(
        documents=["Chunk 1.", "Chunk 2."],
        metadata=[
            {"course_title": "AI Basics", "lesson_number": 1},
            {"course_title": "AI Basics", "lesson_number": 2},
        ],
    )

    tool.execute(query="AI")

    assert len(tool.last_sources) == 2
