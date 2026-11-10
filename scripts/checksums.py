"""SHA-256 manifest of input data and saved results (cross-platform; no sha256sum needed).

  python scripts/checksums.py write                 # (re)write CHECKSUMS.sha256 for data/ and results/
  python scripts/checksums.py verify                # exit 1 if any file is changed, missing or unlisted
  python scripts/checksums.py compare DIR_A DIR_B   # compare two result trees, e.g. results vs a re-run

Text files (.json .jsonl .md .txt .csv) are hashed with CRLF normalised to LF, so outputs written on
Windows and on Linux compare equal. Binary files (.pkl .png) are hashed byte-for-byte; pickles and
PNGs are only expected to match with the pinned library versions in requirements.txt.
"""
import hashlib
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (cwd = repo root)

MANIFEST = Path("CHECKSUMS.sha256")
TREES = ("data", "results")
TEXT = {".json", ".jsonl", ".md", ".txt", ".csv"}


def digest(p: Path) -> str:
    b = p.read_bytes()
    if p.suffix in TEXT:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def files(root: Path):
    return sorted(q for q in root.rglob("*") if q.is_file() and "__pycache__" not in q.parts)


def table(roots, base: Path = Path(".")) -> dict[str, str]:
    """Paths are recorded relative to `base` (the repo root), e.g. 'results/E2_budget.json'."""
    return {q.relative_to(base).as_posix(): digest(q) for r in roots for q in files(base / r)}


def verify(base: Path = Path("."), out=print) -> int:
    """0 only if every listed file matches AND no file under data/ or results/ is unlisted.
    Changed, missing or unlisted files all give exit code 1."""
    want = dict(reversed(line.split("  ", 1)) for line in (base / MANIFEST).read_text(encoding="utf-8").splitlines() if line)
    got = table(TREES, base)
    bad = sorted(p for p in want if got.get(p) != want[p])
    new = sorted(set(got) - set(want))
    for p in bad:
        out(("MISSING " if p not in got else "CHANGED ") + p)
    for p in new:
        out("UNLISTED " + p)
    out(f"{len(want) - len(bad)}/{len(want)} listed files match" + (f"; {len(new)} unlisted file(s)" if new else ""))
    return 1 if bad or new else 0


def main(argv):
    mode = argv[1] if len(argv) > 1 else "verify"
    if mode == "write":
        t = table(TREES)
        MANIFEST.write_text("".join(f"{h}  {p}\n" for p, h in t.items()), encoding="utf-8", newline="\n")
        print(f"wrote {len(t)} entries to {MANIFEST}")
        return 0
    if mode == "verify":
        return verify()
    if mode == "compare":
        a, b = Path(argv[2]), Path(argv[3])
        ta = {q.relative_to(a).as_posix(): digest(q) for q in files(a)}
        tb = {q.relative_to(b).as_posix(): digest(q) for q in files(b)}
        same = sorted(k for k in ta if tb.get(k) == ta[k])
        for k in sorted(set(ta) | set(tb)):
            if k not in same:
                print(("ONLY-IN-A " if k not in tb else "ONLY-IN-B " if k not in ta else "DIFFERS   ") + k)
        print(f"{len(same)}/{len(set(ta) | set(tb))} files identical")
        return 0 if len(same) == len(set(ta) | set(tb)) else 1
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
