# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

Requires Python 3.13+ and `uv`. Always use `uv` for all dependency management and script execution — never use `pip` or bare `python`/`uvicorn` directly. This includes running Python files (`uv run python file.py`) and adding dependencies (`uv add <package>`).

Create a `.env` file in the project root with `ANTHROPIC_API_KEY=...` before starting.

```bash
# From project root (Git Bash on Windows)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

App runs at `http://localhost:8000`. There are no tests or linting scripts configured.

Install dependencies:
```bash
uv sync
```

## Architecture

This is a RAG chatbot that answers questions about course materials using Claude's tool-use feature.

**Request flow:**
1. Browser `POST /api/query` → `app.py` → `RAGSystem.query()`
2. `RAGSystem` attaches conversation history and passes the `search_course_content` tool definition to `AIGenerator`
3. `AIGenerator` makes a first Claude API call. If Claude decides to search, it returns `stop_reason="tool_use"`
4. `CourseSearchTool` executes the search against ChromaDB, resolving fuzzy course names via vector similarity on `course_catalog`, then querying `course_content`
5. Results are appended as a `tool_result` message and a second Claude call produces the final answer
6. Sources and response are returned to the browser

**Component responsibilities:**
- `rag_system.py` — orchestrator; wires all components; handles document ingestion deduplication
- `ai_generator.py` — all Claude API calls; owns the two-turn tool-use loop
- `vector_store.py` — ChromaDB wrapper; two collections: `course_catalog` (one doc per course) and `course_content` (chunked text); course name resolution uses vector search on `course_catalog`
- `document_processor.py` — parses course `.txt` files, splits into sentence-aware overlapping chunks (800 chars, 100 overlap), prepends lesson context to each chunk
- `search_tools.py` — defines the `search_course_content` Anthropic tool and formats results; `ToolManager` is extensible (register additional `Tool` subclasses)
- `session_manager.py` — in-memory conversation history keyed by session ID; retains last 2 exchanges

**Key config** (`backend/config.py`):
- Model: `claude-sonnet-4-20250514`
- Embedding model: `all-MiniLM-L6-v2` (via `sentence-transformers`)
- ChromaDB stored at `backend/chroma_db/`
- `MAX_HISTORY = 2` exchanges retained per session

## Course Document Format

Files in `docs/` must follow this structure for `DocumentProcessor` to parse them correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <lesson title>
...
```

Documents are loaded once on startup and deduplicated by course title — re-adding a course with the same title is a no-op unless `clear_existing=True` is passed.
