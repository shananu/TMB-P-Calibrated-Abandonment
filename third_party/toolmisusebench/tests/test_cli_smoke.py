from __future__ import annotations

import json

from toolmisusebench.cli.main import main


def test_cli_generate_smoke(tmp_path) -> None:
    out_dir = tmp_path / "v0_1"
    exit_code = main(
        [
            "generate",
            "--version",
            "v0.1",
            "--out",
            str(out_dir),
            "--seed",
            "42",
            "--no-parquet",
        ]
    )
    assert exit_code == 0
    assert (out_dir / "train" / "tasks.jsonl").exists()
    assert (out_dir / "dev" / "tasks.jsonl").exists()
    assert (out_dir / "test_public" / "tasks.jsonl").exists()
    assert (out_dir / "manifest.json").exists()


def test_cli_generate_with_custom_split_sizes(tmp_path) -> None:
    out_dir = tmp_path / "custom"
    exit_code = main(
        [
            "generate",
            "--version",
            "v0.1",
            "--out",
            str(out_dir),
            "--seed",
            "42",
            "--no-parquet",
            "--train-size",
            "40",
            "--dev-size",
            "8",
            "--test-size",
            "12",
        ]
    )
    assert exit_code == 0
    train_lines = (out_dir / "train" / "tasks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    dev_lines = (out_dir / "dev" / "tasks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    test_lines = (out_dir / "test_public" / "tasks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == 40
    assert len(dev_lines) == 8
    assert len(test_lines) == 12


def test_cli_generate_rejects_partial_split_override(tmp_path) -> None:
    out_dir = tmp_path / "invalid"
    exit_code = main(
        [
            "generate",
            "--version",
            "v0.1",
            "--out",
            str(out_dir),
            "--seed",
            "42",
            "--no-parquet",
            "--train-size",
            "40",
        ]
    )
    assert exit_code == 2


def test_cli_eval_smoke(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset" / "test_public"
    dataset_dir.mkdir(parents=True)
    task_path = dataset_dir / "tasks.jsonl"

    task = {
        "task_id": "cli_eval_1",
        "split": "test_public",
        "difficulty": "easy",
        "domain": "crud",
        "instruction": "Close account a1",
        "toolset_id": "crud_v1",
        "tool_schemas": [],
        "initial_state": {"accounts": {"a1": {"id": "a1", "status": "active"}}},
        "success_criteria": [{"op": "equals", "path": "accounts.a1.status", "value": "closed"}],
        "budget": {"max_steps": 4, "max_tool_calls": 4, "max_retries": 1, "timeout_ms": 1000},
        "fault_plan": [],
        "seed": 1,
    }
    task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")

    report_path = tmp_path / "out" / "report.json"
    exit_code = main(
        [
            "eval",
            "--dataset",
            str(tmp_path / "dataset"),
            "--split",
            "test_public",
            "--agent",
            "heuristic",
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0
    assert report_path.exists()
