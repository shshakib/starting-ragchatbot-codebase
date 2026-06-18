#!/bin/bash
# Format the codebase with black.
set -e
cd "$(dirname "$0")/.."

echo "Running black..."
uv run black .
