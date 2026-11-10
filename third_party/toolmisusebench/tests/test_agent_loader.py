from __future__ import annotations

import json
import sys

from toolmisusebench.cli.main import main


def test_cli_eval_supports_custom_agent_module(tmp_path) -> None:
    module_path = tmp_path / "custom_agent.py"
    module_path.write_text(
        "\n".join(
            [
                "from toolmisusebench.types import Action",
                "",
                "class MyAgent:",
                "    def __init__(self, tool_name='list_jobs'):",
                "        self.tool_name = tool_name",
                "    def reset(self):",
                "        pass",
                "    def act(self, observation):",
                "        return Action(tool_name=self.tool_name, args={})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset_dir = tmp_path / "dataset" / "test_public"
    dataset_dir.mkdir(parents=True)
    task_path = dataset_dir / "tasks.jsonl"
    task = {
        "task_id": "custom_agent_eval",
        "split": "test_public",
        "difficulty": "easy",
        "domain": "scheduling",
        "instruction": "List jobs",
        "toolset_id": "scheduler_v1",
        "tool_schemas": [],
        "initial_state": {"jobs": []},
        "success_criteria": [],
        "budget": {"max_steps": 2, "max_tool_calls": 2, "max_retries": 1, "timeout_ms": 1000},
        "fault_plan": [],
        "seed": 1,
    }
    task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")

    sys.path.insert(0, str(tmp_path))
    try:
        report_path = tmp_path / "out" / "report.json"
        exit_code = main(
            [
                "eval",
                "--dataset",
                str(tmp_path / "dataset"),
                "--split",
                "test_public",
                "--agent-module",
                "custom_agent:MyAgent",
                "--agent-kwargs",
                '{"tool_name": "list_jobs"}',
                "--report",
                str(report_path),
            ]
        )
    finally:
        sys.path.pop(0)

    assert exit_code == 0
    assert report_path.exists()


def test_cli_eval_rejects_invalid_agent_kwargs(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset" / "test_public"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "tasks.jsonl").write_text("", encoding="utf-8")

    exit_code = main(
        [
            "eval",
            "--dataset",
            str(tmp_path / "dataset"),
            "--split",
            "test_public",
            "--agent-module",
            "nope:Agent",
            "--agent-kwargs",
            "not_json",
            "--report",
            str(tmp_path / "out" / "report.json"),
        ]
    )
    assert exit_code == 2
