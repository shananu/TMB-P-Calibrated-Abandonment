"""Regression tests for scripts/checksums.py verify (exit code must be nonzero for changed, missing
or UNLISTED files). Run: python -m tests test_checksums"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import tests

sys.path.insert(0, str(tests.ROOT / "scripts"))  # checksums.py imports _bootstrap
_spec = importlib.util.spec_from_file_location("checksums", tests.ROOT / "scripts" / "checksums.py")
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)


def _tree(tmp: Path) -> Path:
    for rel, text in (("data/a.json", '{"x": 1}\r\n'), ("results/b.md", "hello\n"), ("results/c.png", "\x89PNG")):
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_bytes(text.encode("latin-1"))
    t = ck.table(ck.TREES, tmp)
    (tmp / ck.MANIFEST).write_text("".join(f"{h}  {p}\n" for p, h in t.items()), encoding="utf-8")
    return tmp


def test_verify_clean_tree_passes_and_crlf_is_normalised():
    with tempfile.TemporaryDirectory() as d:
        base = _tree(Path(d))
        assert ck.verify(base, out=lambda *_: None) == 0
        (base / "data/a.json").write_bytes(b'{"x": 1}\n')  # CRLF -> LF only: still equal
        assert ck.verify(base, out=lambda *_: None) == 0


def test_verify_fails_on_unlisted_file():
    with tempfile.TemporaryDirectory() as d:
        base = _tree(Path(d))
        (base / "results/extra.json").write_text("{}", encoding="utf-8")
        msgs = []
        assert ck.verify(base, out=msgs.append) == 1
        assert "UNLISTED results/extra.json" in msgs


def test_verify_fails_on_changed_and_missing_file():
    with tempfile.TemporaryDirectory() as d:
        base = _tree(Path(d))
        (base / "results/b.md").write_text("changed\n", encoding="utf-8")
        assert ck.verify(base, out=lambda *_: None) == 1
    with tempfile.TemporaryDirectory() as d:
        base = _tree(Path(d))
        (base / "results/c.png").unlink()
        msgs = []
        assert ck.verify(base, out=msgs.append) == 1 and "MISSING results/c.png" in msgs
