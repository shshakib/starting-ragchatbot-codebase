import sys
import os

# Make backend/ importable so that "from vector_store import ..." works in tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
