from __future__ import annotations

import json

from toolmisusebench.harness.artifacts import write_evaluation_artifacts
from toolmisusebench.harness.metrics import episode_to_trace_row, task_metrics
from toolmisusebench.harness.runner import run_episode
from toolmisusebench.harness.scoring import score_episodes
from toolmisusebench.types import Action, Budget, Observation, Task


class FixedAgent:
    def __init__(self, actions: list[Action]) -> None:
        self._base_actions = actions
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def act(self, observation: Observation) -> Action | None:
        if self._cursor >= len(self._base_actions):
            return None
        action = self._base_actions[self._cursor]
        self._cursor += 1
        return action


def test_run_episode_and_scoring_success_path() -> None:
    task = Task(
        task_id="harness_success",
        split="dev",
        difficulty="easy",
        domain="crud",
        instruction="Close account a1",
        toolset_id="crud_v1",
        tool_schemas=[],
        initial_state={"accounts": {"a1": {"id": "a1", "status": "active"}}},
        success_criteria=[{"op": "equals", "path": "accounts.a1.status", "value": "closed"}],
        budget=Budget(max_steps=5, max_tool_calls=5, max_retries=1, timeout_ms=1000),
        fault_plan=[],
        seed=4,
    )
    agent = FixedAgent([Action(tool_name="close_account", args={"id": "a1", "reason": "requested"})])

    result = run_episode(task, agent)
    assert result.success is True
    assert result.invalid_calls == 0
    assert result.tool_calls == 1

    metrics = task_metrics(result)
    assert metrics["TaskSuccess"] == 1
    assert metrics["ToolCallsUsed"] == 1

    scored = score_episodes([result])
    assert scored["aggregate"]["TaskSuccess"] == 1.0
    assert scored["budgeted_success"]["auc"] == 1.0


def test_recovery_metrics_and_artifact_writes(tmp_path) -> None:
    task = Task(
        task_id="harness_recovery",
        split="dev",
        difficulty="easy",
        domain="retrieval",
        instruction="Fetch document d1",
        toolset_id="retrieval_v1",
        tool_schemas=[],
        initial_state={"docs": [{"id": "d1", "title": "Doc", "text": "hello"}]},
        success_criteria=[{"op": "equals", "path": "docs.0.id", "value": "d1"}],
        budget=Budget(max_steps=5, max_tool_calls=5, max_retries=1, timeout_ms=1000),
        fault_plan=[
            {
                "fault_type": "timeout",
                "trigger": {"on_tool": "get_doc", "on_nth_call": 1},
                "severity": "soft",
                "payload": {"timeout_ms": 25},
            }
        ],
        seed=7,
    )
    agent = FixedAgent(
        [
            Action(tool_name="get_doc", args={"doc_id": "d1"}),
            Action(tool_name="get_doc", args={"doc_id": "d1"}),
        ]
    )

    result = run_episode(task, agent)
    assert result.fault_seen is True
    assert result.recovery_success is True
    assert result.time_to_recovery == 1

    scored = score_episodes([result])
    trace_rows = [episode_to_trace_row(result)]
    report_path = tmp_path / "out" / "report.json"
    paths = write_evaluation_artifacts(report_path, scored, trace_rows)

    assert (tmp_path / "out" / "report.json").exists()
    assert (tmp_path / "out" / "report.traces.jsonl").exists()

    report_payload = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report_payload["aggregate"]["RecoverySuccess"] == 1.0
    assert paths["report"].endswith("report.json")


def test_retry_budget_is_enforced() -> None:
    task = Task(
        task_id="harness_retry_budget",
        split="dev",
        difficulty="easy",
        domain="retrieval",
        instruction="Try get_doc repeatedly",
        toolset_id="retrieval_v1",
        tool_schemas=[],
        initial_state={"docs": [{"id": "d1", "title": "Doc", "text": "hello"}]},
        success_criteria=[],
        budget=Budget(max_steps=5, max_tool_calls=5, max_retries=1, timeout_ms=1000),
        fault_plan=[],
        seed=5,
    )

    agent = FixedAgent(
        [
            Action(tool_name="get_doc", args={"doc_id": 1}),
            Action(tool_name="get_doc", args={"doc_id": 1}),
        ]
    )

    result = run_episode(task, agent)
    assert result.budget_exceeded is True
    assert result.catastrophic_failure is True
