from __future__ import annotations

import json

from toolmisusebench.baselines import build_agent
from toolmisusebench.harness.runner import run_episode
from toolmisusebench.types import Budget, Task
from toolmisusebench.version2 import analyze_dataset_v2, generate_dataset_v2


def test_v2_generator_crosses_domain_fault_difficulty_and_budget(tmp_path) -> None:
    out = tmp_path / "v0_2"
    generate_dataset_v2(
        version="v0.2",
        out_dir=out,
        seed=42,
        split_sizes={"train": 216, "dev": 216, "test_public": 216},
    )

    report = analyze_dataset_v2(out, ["test_public"])
    summary = report["per_split"]["test_public"]

    assert summary["rows"] == 216
    assert summary["duplicate_task_ids"] == 0
    assert set(summary["domain_counts"]) == {"crud", "retrieval", "files", "scheduling"}
    assert set(summary["fault_counts"]) == {
        "clean",
        "timeout",
        "rate_limit",
        "schema_drift",
        "authz",
        "adversarial_error",
    }
    assert set(summary["difficulty_counts"]) == {"easy", "medium", "hard"}
    assert len(summary["budget_counts"]) == 3

    for domain, fault_counts in summary["domain_fault_matrix"].items():
        assert domain in {"crud", "retrieval", "files", "scheduling"}
        assert set(fault_counts) == set(summary["fault_counts"])

    assert summary["output_grounded_task_count"] > 0


def test_cli_v2_generation_writes_freeze(tmp_path) -> None:
    from toolmisusebench.cli.main import main

    out = tmp_path / "cli_v2"
    code = main(
        [
            "generate",
            "--version",
            "v0.2",
            "--out",
            str(out),
            "--seed",
            "7",
            "--train-size",
            "12",
            "--dev-size",
            "12",
            "--test-size",
            "12",
        ]
    )

    assert code == 0
    assert (out / "v0_2_freeze.json").exists()
    freeze = json.loads((out / "v0_2_freeze.json").read_text(encoding="utf-8"))
    assert freeze["design"] == "crossed_domain_fault_difficulty_budget"


def test_instruction_parser_baseline_solves_clean_crud_task() -> None:
    task = Task(
        task_id="v2_parser_clean_crud",
        split="test_public",
        difficulty="easy",
        domain="crud",
        instruction="Archive account at0000 by closing it with reason risk_review.",
        toolset_id="crud_v1",
        tool_schemas=[],
        initial_state={"accounts": {"at0000": {"id": "at0000", "status": "active"}}},
        success_criteria=[
            {"op": "equals", "path": "accounts.at0000.status", "value": "closed"},
            {"op": "equals", "path": "accounts.at0000.closed_reason", "value": "risk_review"},
        ],
        budget=Budget(max_steps=4, max_tool_calls=4, max_retries=1, timeout_ms=1000),
        fault_plan=[],
        seed=1,
    )

    result = run_episode(task, build_agent("instruction_parser"))
    assert result.success is True
    assert result.tool_calls == 1

