"""Test package. Importing it puts the repo root and vendored ToolMisuseBench on sys.path and makes
the repo root the working directory, so `python -m tests` works on Windows and POSIX without PYTHONPATH."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "third_party" / "toolmisusebench", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(ROOT)
