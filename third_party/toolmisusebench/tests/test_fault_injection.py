from __future__ import annotations

from toolmisusebench.envs import CrudEnv, RetrievalEnv
from toolmisusebench.types import Action, Budget, Task


def _retrieval_task(fault_plan: list[dict]) -> Task:
    return Task(
        task_id="retrieval_fault_test",
        split="dev",
        difficulty="easy",
        domain="retrieval",
        instruction="test",
        toolset_id="retrieval_v1",
        initial_state={"docs": [{"id": "d1", "title": "Doc", "text": "hello"}]},
        success_criteria=[],
        budget=Budget(max_steps=10, max_tool_calls=10, max_retries=1, timeout_ms=1000),
        fault_plan=fault_plan,
        seed=13,
    )


def test_timeout_trigger_on_nth_call() -> None:
    task = _retrieval_task(
        [
            {
                "fault_type": "timeout",
                "trigger": {"on_tool": "get_doc", "on_nth_call": 2},
                "severity": "soft",
                "payload": {"timeout_ms": 50},
            }
        ]
    )

    env = RetrievalEnv()
    env.reset(task)

    first = env.step(Action(tool_name="get_doc", args={"doc_id": "d1"}))
    second = env.step(Action(tool_name="get_doc", args={"doc_id": "d1"}))

    assert first.error is None
    assert second.error is not None
    assert second.error.code == "timeout"
    assert second.error.details["timeout_ms"] == 50


def test_schema_drift_rename_arg_breaks_call() -> None:
    task = _retrieval_task(
        [
            {
                "fault_type": "schema_drift",
                "trigger": {"on_tool": "get_doc", "on_nth_call": 1},
                "severity": "hard",
                "payload": {"rename_args": {"doc_id": "id"}},
            }
        ]
    )

    env = RetrievalEnv()
    env.reset(task)

    result = env.step(Action(tool_name="get_doc", args={"doc_id": "d1"}))
    assert result.error is not None
    assert result.error.code == "missing_required_arg"
    assert any(event["fault_type"] == "schema_drift" for event in result.error.details.get("faults", []))


def test_authz_rate_limit_and_prob_are_deterministic() -> None:
    task = Task(
        task_id="crud_fault_test",
        split="dev",
        difficulty="easy",
        domain="crud",
        instruction="test",
        toolset_id="crud_v1",
        initial_state={"accounts": {"a1": {"id": "a1", "status": "active"}}},
        success_criteria=[],
        budget=Budget(max_steps=10, max_tool_calls=10, max_retries=1, timeout_ms=1000),
        fault_plan=[
            {
                "fault_type": "authz",
                "trigger": {"on_tool": "close_account", "when_arg_matches": {"reason": "fraud"}},
                "severity": "hard",
                "payload": {"deny": True},
            },
            {
                "fault_type": "rate_limit",
                "trigger": {"on_tool": "get_account", "prob": {"p": 0.5, "seed": 99}},
                "severity": "soft",
                "payload": {"after_n_calls": 0, "retry_after_ms": 123},
            },
        ],
        seed=42,
    )

    env1 = CrudEnv()
    env2 = CrudEnv()
    env1.reset(task)
    env2.reset(task)

    auth1 = env1.step(Action(tool_name="close_account", args={"id": "a1", "reason": "fraud"}))
    auth2 = env2.step(Action(tool_name="close_account", args={"id": "a1", "reason": "fraud"}))
    assert auth1.model_dump() == auth2.model_dump()
    assert auth1.error is not None
    assert auth1.error.code == "authz_denied"

    rl11 = env1.step(Action(tool_name="get_account", args={"id": "a1"}))
    rl12 = env1.step(Action(tool_name="get_account", args={"id": "a1"}))
    rl21 = env2.step(Action(tool_name="get_account", args={"id": "a1"}))
    rl22 = env2.step(Action(tool_name="get_account", args={"id": "a1"}))

    assert rl11.model_dump() == rl21.model_dump()
    assert rl12.model_dump() == rl22.model_dump()


def test_adversarial_transforms_error_message() -> None:
    task = _retrieval_task(
        [
            {
                "fault_type": "timeout",
                "trigger": {"on_tool": "get_doc", "on_nth_call": 1},
                "severity": "soft",
                "payload": {},
            },
            {
                "fault_type": "adversarial_error",
                "trigger": {"on_tool": "get_doc", "on_nth_call": 1},
                "severity": "soft",
                "payload": {"message": "Operation failed unexpectedly.", "code": "ambiguous_error"},
            },
        ]
    )

    env = RetrievalEnv()
    env.reset(task)

    result = env.step(Action(tool_name="get_doc", args={"doc_id": "d1"}))
    assert result.error is not None
    assert result.error.code == "ambiguous_error"
    assert result.error.message == "Operation failed unexpectedly."
    assert result.error.details.get("original_error_code") == "timeout"
