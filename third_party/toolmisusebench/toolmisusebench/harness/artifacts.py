from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_evaluation_artifacts(
    report_path: str | Path,
    scored_payload: dict[str, Any],
    trace_rows: list[dict[str, Any]],
) -> dict[str, str]:
    report = Path(report_path)
    trace_path = report.with_suffix(".traces.jsonl")
    write_json(report, scored_payload)
    write_jsonl(trace_path, trace_rows)
    return {
        "report": str(report),
        "traces": str(trace_path),
    }
