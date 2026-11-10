"""Run every test_* function in every tests/test_*.py module: `python -m tests` (or `python -m tests test_shift`).
Prints PASS/FAIL per test and exits non-zero if any test fails. Also works under pytest."""
import importlib
import sys
import time
import traceback
from pathlib import Path

import tests  # noqa: F401  (sets paths and cwd)

mods = sys.argv[1:] or sorted(p.stem for p in Path(__file__).parent.glob("test_*.py"))
failed, n = [], 0
for mod_name in mods:
    mod = importlib.import_module(f"tests.{mod_name}")
    for name in sorted(k for k in vars(mod) if k.startswith("test_") and callable(getattr(mod, k))):
        n += 1
        t0 = time.time()
        try:
            getattr(mod, name)()
            print(f"PASS {mod_name}.{name} ({time.time() - t0:.1f}s)", flush=True)
        except Exception:
            failed.append(f"{mod_name}.{name}")
            print(f"FAIL {mod_name}.{name}\n{traceback.format_exc()}", flush=True)
print(f"\n{n - len(failed)}/{n} tests passed" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
sys.exit(1 if failed else 0)
