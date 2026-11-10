"""Shared setup imported first by every script (`import _bootstrap`).

* Puts the repository root and the vendored ToolMisuseBench on sys.path, so no PYTHONPATH is needed
  (the separator differs between Windows ';' and POSIX ':').
* Changes the working directory to the repository root, so `data/...` paths resolve from anywhere.
* RESULTS: output directory, `results/` by default. Set TMBP_RESULTS_DIR to regenerate into a
  separate directory and compare against the reference outputs (see scripts/checksums.py).
* guard(): refuses to overwrite existing outputs unless TMBP_OVERWRITE=1.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "third_party" / "toolmisusebench", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
except AttributeError:
    pass

RESULTS = Path(os.environ.get("TMBP_RESULTS_DIR", "results"))
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(exist_ok=True)


def guard(*paths: Path) -> None:
    """Exit before doing any work if an output already exists (set TMBP_OVERWRITE=1 to allow)."""
    if os.environ.get("TMBP_OVERWRITE") == "1":
        return
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        sys.exit("refusing to overwrite existing output(s): " + ", ".join(existing)
                 + "\nset TMBP_RESULTS_DIR to write elsewhere, or TMBP_OVERWRITE=1 to replace them")
