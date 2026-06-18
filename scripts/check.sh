#!/bin/bash
# Run code quality checks (formatting + tests) without modifying files.
set -e
cd "$(dirname "$0")/.."

echo "Checking black formatting..."
uv run black --check --diff .

echo "Running tests..."
uv run pytest backend/tests -q
