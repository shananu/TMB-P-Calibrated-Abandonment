from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def analyze_dataset_v2(dataset: str | Path, splits: list[str] | None = None) -> dict[str, Any]:
    dataset_path = Path(dataset)
    target_splits = splits or _discover_splits(dataset_path)
    per_split: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for split in target_splits:
        rows = _read_split_rows(dataset_path, split)
        all_rows.extend(rows)
        per_split[split] = _summarize_rows(rows)

    return {
        "dataset": str(dataset_path),
        "splits": target_splits,
        "overall": _summarize_rows(all_rows),
        "per_split": per_split,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "duplicate_task_ids": 0,
            "domain_counts": {},
            "fault_counts": {},
            "difficulty_counts": {},
            "budget_counts": {},
            "domain_fault_matrix": {},
            "fault_difficulty_matrix": {},
            "domain_difficulty_matrix": {},
            "criterion_op_counts": {},
            "output_grounded_task_count": 0,
            "recoverability_counts": {},
            "instruction_uniqueness_ratio": 0.0,
        }

    ids = [str(row.get("task_id", "")) for row in rows]
    instruction_count = len({str(row.get("instruction", "")) for row in rows})

    return {
        "rows": len(rows),
        "duplicate_task_ids": len(rows) - len(set(ids)),
        "domain_counts": dict(Counter(str(row.get("domain", "unknown")) for row in rows)),
        "fault_counts": dict(Counter(_primary_fault(row) for row in rows)),
        "difficulty_counts": dict(Counter(str(row.get("difficulty", "unknown")) for row in rows)),
        "budget_counts": dict(Counter(_budget_key(row) for row in rows)),
        "domain_fault_matrix": _matrix(rows, "domain", _primary_fault),
        "fault_difficulty_matrix": _matrix(rows, _primary_fault, "difficulty"),
        "domain_difficulty_matrix": _matrix(rows, "domain", "difficulty"),
        "criterion_op_counts": dict(Counter(_criterion_ops(rows))),
        "output_grounded_task_count": sum(1 for row in rows if _has_output_grounded_criterion(row)),
        "recoverability_counts": dict(Counter(_recoverability(row) for row in rows)),
        "instruction_uniqueness_ratio": round(instruction_count / len(rows), 6),
    }


def _discover_splits(dataset_path: Path) -> list[str]:
    return [name for name in ("train", "dev", "test_public") if (dataset_path / name).is_dir()]


def _read_split_rows(dataset_path: Path, split: str) -> list[dict[str, Any]]:
    path = dataset_path / split / "tasks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing v2 split file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _primary_fault(row: dict[str, Any]) -> str:
    fault_plan = row.get("fault_plan", [])
    if not fault_plan:
        return "clean"
    first = fault_plan[0]
    if not isinstance(first, dict):
        return "unknown"
    return str(first.get("fault_type", "unknown"))


def _budget_key(row: dict[str, Any]) -> str:
    budget = row.get("budget", {})
    return (
        f"steps={budget.get('max_steps')};"
        f"calls={budget.get('max_tool_calls')};"
        f"retries={budget.get('max_retries')}"
    )


def _matrix(
    rows: list[dict[str, Any]],
    row_key: str | Any,
    col_key: str | Any,
) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        left = row_key(row) if callable(row_key) else str(row.get(row_key, "unknown"))
        right = col_key(row) if callable(col_key) else str(row.get(col_key, "unknown"))
        matrix[str(left)][str(right)] += 1
    return {key: dict(value) for key, value in sorted(matrix.items())}


def _criterion_ops(rows: list[dict[str, Any]]) -> list[str]:
    ops: list[str] = []
    for row in rows:
        for criterion in row.get("success_criteria", []):
            if isinstance(criterion, dict):
                ops.append(str(criterion.get("op", "unknown")))
    return ops


def _has_output_grounded_criterion(row: dict[str, Any]) -> bool:
    return any(
        isinstance(criterion, dict) and str(criterion.get("op", "")).startswith("tool_output_")
        for criterion in row.get("success_criteria", [])
    )


def _recoverability(row: dict[str, Any]) -> str:
    fault_plan = row.get("fault_plan", [])
    if not fault_plan:
        return "clean"
    for fault in fault_plan:
        if not isinstance(fault, dict):
            continue
        payload = fault.get("payload", {})
        if isinstance(payload, dict) and payload.get("recoverability"):
            return str(payload["recoverability"])
    return "unspecified"

