from __future__ import annotations

import json
from pathlib import Path

from toolmisusebench.types import Task


def task_files(dataset_path: Path, split: str) -> list[Path]:
    split_dir = dataset_path / split
    target = split_dir if split_dir.is_dir() else dataset_path
    files = sorted(target.glob("*.jsonl")) + sorted(target.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No task files found in '{target}'. Expected .jsonl or .json files.")
    return files


def load_tasks(dataset: str | Path, split: str) -> list[Task]:
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    tasks: list[Task] = []
    for file_path in task_files(dataset_path, split):
        if file_path.suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        tasks.append(Task.model_validate(json.loads(stripped)))
        else:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                tasks.extend(Task.model_validate(item) for item in raw)
            else:
                tasks.append(Task.model_validate(raw))
    return [task for task in tasks if task.split == split]
