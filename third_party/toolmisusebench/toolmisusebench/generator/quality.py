from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def analyze_dataset(dataset: str | Path, splits: list[str] | None = None) -> dict[str, Any]:
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    target_splits = splits or _discover_splits(dataset_path)
    if not target_splits:
        raise ValueError("No dataset splits found.")

    per_split: dict[str, dict[str, Any]] = {}
    overall_rows = 0
    overall_domains: Counter[str] = Counter()
    overall_faults: Counter[str] = Counter()

    for split in target_splits:
        rows = _read_split_rows(dataset_path, split)
        summary = _summarize_rows(rows)
        per_split[split] = summary
        overall_rows += summary["rows"]
        overall_domains.update(summary["domain_counts"])
        overall_faults.update(summary["primary_fault_counts"])

    return {
        "dataset": str(dataset_path),
        "splits": target_splits,
        "overall": {
            "rows": overall_rows,
            "domain_counts": dict(overall_domains),
            "primary_fault_counts": dict(overall_faults),
        },
        "per_split": per_split,
    }


def _discover_splits(dataset_path: Path) -> list[str]:
    expected = ["train", "dev", "test_public"]
    found = [name for name in expected if (dataset_path / name).is_dir()]
    if found:
        return found
    return ["all"]


def _read_split_rows(dataset_path: Path, split: str) -> list[dict[str, Any]]:
    split_dir = dataset_path if split == "all" else dataset_path / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split path does not exist: {split_dir}")

    files = sorted(split_dir.glob("*.jsonl")) + sorted(split_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No task files found under split '{split}' in {split_dir}")

    rows: list[dict[str, Any]] = []
    for file_path in files:
        if file_path.suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        rows.append(json.loads(stripped))
        else:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                rows.extend(raw)
            elif isinstance(raw, dict):
                rows.append(raw)
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "unique_task_ids": 0,
            "duplicate_task_ids": 0,
            "domain_counts": {},
            "primary_fault_counts": {},
            "unique_instructions": 0,
            "unique_initial_states": 0,
            "unique_success_criteria": 0,
            "instruction_uniqueness_ratio": 0.0,
        }

    ids = [str(row.get("task_id", "")) for row in rows]
    domain_counts = Counter(str(row.get("domain", "unknown")) for row in rows)
    primary_fault_counts = Counter(_primary_fault(row) for row in rows)

    unique_instructions = len({str(row.get("instruction", "")) for row in rows})
    unique_initial_states = len({json.dumps(row.get("initial_state", {}), sort_keys=True) for row in rows})
    unique_success_criteria = len({json.dumps(row.get("success_criteria", []), sort_keys=True) for row in rows})

    return {
        "rows": len(rows),
        "unique_task_ids": len(set(ids)),
        "duplicate_task_ids": len(rows) - len(set(ids)),
        "domain_counts": dict(domain_counts),
        "primary_fault_counts": dict(primary_fault_counts),
        "unique_instructions": unique_instructions,
        "unique_initial_states": unique_initial_states,
        "unique_success_criteria": unique_success_criteria,
        "instruction_uniqueness_ratio": round(unique_instructions / len(rows), 6),
    }


def _primary_fault(row: dict[str, Any]) -> str:
    fault_plan = row.get("fault_plan", [])
    if not isinstance(fault_plan, list) or not fault_plan:
        return "clean"
    first = fault_plan[0]
    if not isinstance(first, dict):
        return "unknown"
    return str(first.get("fault_type", "unknown"))
