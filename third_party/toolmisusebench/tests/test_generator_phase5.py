from __future__ import annotations

import json

from toolmisusebench.generator import generate_dataset
from toolmisusebench.generator.core import _validate_task_coherence
from toolmisusebench.types import Budget, Task


def _read(path):
    return path.read_text(encoding="utf-8")


def test_generate_dataset_writes_split_artifacts_and_freeze(tmp_path) -> None:
    out = tmp_path / "data" / "v0_1"
    payload = generate_dataset(version="v0.1", out_dir=out, seed=42, write_parquet=False)

    assert payload["splits"] == {"train": 24, "dev": 12, "test_public": 12}
    assert (out / "train" / "tasks.jsonl").exists()
    assert (out / "dev" / "tasks.jsonl").exists()
    assert (out / "test_public" / "tasks.jsonl").exists()
    assert (out / "manifest.json").exists()
    assert (out / "v0_1_freeze.json").exists()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "v0.1"
    assert len(manifest["files"]) == 3
    assert payload["quality_issues"] == 0


def test_generate_dataset_is_deterministic_for_same_seed(tmp_path) -> None:
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    generate_dataset(version="v0.1", out_dir=out1, seed=7, write_parquet=False)
    generate_dataset(version="v0.1", out_dir=out2, seed=7, write_parquet=False)

    for split in ("train", "dev", "test_public"):
        assert _read(out1 / split / "tasks.jsonl") == _read(out2 / split / "tasks.jsonl")


def test_generate_dataset_supports_large_profile(tmp_path) -> None:
    out = tmp_path / "large"
    payload = generate_dataset(version="v0.1", out_dir=out, seed=9, write_parquet=False, size_profile="large")
    assert payload["splits"] == {"train": 5000, "dev": 800, "test_public": 1000}


def test_generate_dataset_supports_split_overrides(tmp_path) -> None:
    out = tmp_path / "custom"
    payload = generate_dataset(
        version="v0.1",
        out_dir=out,
        seed=9,
        write_parquet=False,
        size_profile="small",
        split_sizes={"train": 30, "dev": 10, "test_public": 20},
    )
    assert payload["splits"] == {"train": 30, "dev": 10, "test_public": 20}


def test_generated_tasks_show_instruction_and_state_variability(tmp_path) -> None:
    out = tmp_path / "varied"
    generate_dataset(
        version="v0.1",
        out_dir=out,
        seed=12,
        write_parquet=False,
        split_sizes={"train": 200, "dev": 20, "test_public": 20},
    )

    lines = (out / "train" / "tasks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    unique_instructions = {row["instruction"] for row in records}
    unique_states = {json.dumps(row["initial_state"], sort_keys=True) for row in records}
    unique_criteria = {json.dumps(row["success_criteria"], sort_keys=True) for row in records}

    assert len(unique_instructions) >= 80
    assert len(unique_states) >= 80
    assert len(unique_criteria) >= 8


def test_coherence_validator_flags_incoherent_task() -> None:
    bad_task = Task(
        task_id="bad_task",
        split="dev",
        difficulty="easy",
        domain="retrieval",
        instruction="",
        toolset_id="retrieval_v1",
        tool_schemas=[
            {
                "name": "get_doc",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
                "additionalProperties": False,
            }
        ],
        initial_state={"docs": []},
        success_criteria=[{"op": "successful_tool_call", "tool_name": "missing_tool"}],
        budget=Budget(max_steps=2, max_tool_calls=2, max_retries=1, timeout_ms=1000),
        fault_plan=[],
        seed=1,
    )

    issues = _validate_task_coherence(bad_task)
    assert issues
