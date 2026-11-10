from __future__ import annotations

import json

from toolmisusebench.baselines import PolicyAwareAgent, SchemaRepairAgent
from toolmisusebench.cli.main import main
from toolmisusebench.types import ErrorInfo, Observation, RemainingBudget


def test_schema_repair_fixes_missing_required_arg_with_alias() -> None:
    agent = SchemaRepairAgent()
    agent.reset()

    observation = Observation(
        instruction="test",
        tool_schemas=[
            {
                "name": "get_doc",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
                "additionalProperties": False,
            }
        ],
        transcript=[{"tool_name": "get_doc", "args": {"id": "d1"}, "output": None, "error": None}],
        remaining_budget=RemainingBudget(steps_left=5, tool_calls_left=5, retries_left=1),
        last_error=ErrorInfo(
            code="missing_required_arg",
            message="missing required",
            details={"missing": ["doc_id"]},
        ),
    )

    action = agent.act(observation)
    assert action is not None
    assert action.tool_name == "get_doc"
    assert action.args["doc_id"] == "d1"
    assert "id" not in action.args


def test_schema_repair_casts_invalid_argument_type() -> None:
    agent = SchemaRepairAgent()
    agent.reset()

    observation = Observation(
        instruction="test",
        tool_schemas=[
            {
                "name": "search_docs",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query", "top_k"],
                "additionalProperties": False,
            }
        ],
        transcript=[{"tool_name": "search_docs", "args": {"query": "hello", "top_k": "2"}, "output": None, "error": None}],
        remaining_budget=RemainingBudget(steps_left=5, tool_calls_left=5, retries_left=1),
        last_error=ErrorInfo(
            code="invalid_argument_type",
            message="bad type",
            details={"arg": "top_k", "expected": "integer"},
        ),
    )

    action = agent.act(observation)
    assert action is not None
    assert action.args["top_k"] == 2


def test_policy_aware_blocks_tool_after_authz_feedback() -> None:
    agent = PolicyAwareAgent()
    agent.reset()

    observation = Observation(
        instruction="test",
        tool_schemas=[
            {
                "name": "close_account",
                "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["id", "reason"],
                "additionalProperties": False,
            },
            {
                "name": "get_account",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        ],
        transcript=[
            {
                "tool_name": "close_account",
                "args": {"id": "a1", "reason": "fraud"},
                "output": None,
                "error": {"code": "authz_denied", "message": "denied", "details": {}},
            }
        ],
        remaining_budget=RemainingBudget(steps_left=5, tool_calls_left=5, retries_left=1),
        last_error=ErrorInfo(code="authz_denied", message="denied", details={}),
    )

    action = agent.act(observation)
    assert action is not None
    assert action.tool_name != "close_account"


def test_cli_eval_supports_baseline_agents(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset" / "test_public"
    dataset_dir.mkdir(parents=True)
    task_path = dataset_dir / "tasks.jsonl"
    task = {
        "task_id": "baseline_cli_eval",
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

    schema_report = tmp_path / "out" / "schema_report.json"
    policy_report = tmp_path / "out" / "policy_report.json"

    schema_exit = main(
        [
            "eval",
            "--dataset",
            str(tmp_path / "dataset"),
            "--split",
            "test_public",
            "--agent",
            "schema_repair",
            "--report",
            str(schema_report),
        ]
    )
    policy_exit = main(
        [
            "eval",
            "--dataset",
            str(tmp_path / "dataset"),
            "--split",
            "test_public",
            "--agent",
            "policy_aware",
            "--report",
            str(policy_report),
        ]
    )

    assert schema_exit == 0
    assert policy_exit == 0
    assert schema_report.exists()
    assert policy_report.exists()
