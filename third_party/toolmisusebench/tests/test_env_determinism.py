from __future__ import annotations

from toolmisusebench.envs import CrudEnv, FilesEnv, RetrievalEnv, SchedulerEnv
from toolmisusebench.types import Action, Budget, Task


def _task(domain: str, toolset_id: str, initial_state: dict) -> Task:
    return Task(
        task_id=f"{domain}_determinism",
        split="dev",
        difficulty="easy",
        domain=domain,
        instruction="Determinism check",
        toolset_id=toolset_id,
        tool_schemas=[],
        initial_state=initial_state,
        success_criteria=[],
        budget=Budget(max_steps=10, max_tool_calls=10, max_retries=1, timeout_ms=1000),
        fault_plan=[],
        seed=123,
    )


def test_crud_determinism() -> None:
    task = _task("crud", "crud_v1", {"accounts": {"a1": {"id": "a1", "status": "active"}}})
    env1 = CrudEnv()
    env2 = CrudEnv()

    env1.reset(task)
    env2.reset(task)
    action = Action(tool_name="close_account", args={"id": "a1", "reason": "user_request"})

    result1 = env1.step(action)
    result2 = env2.step(action)

    assert result1.model_dump() == result2.model_dump()
    assert env1.snapshot() == env2.snapshot()


def test_retrieval_determinism() -> None:
    task = _task(
        "retrieval",
        "retrieval_v1",
        {
            "docs": [
                {"id": "d1", "title": "Alpha", "text": "test document alpha"},
                {"id": "d2", "title": "Beta", "text": "test document beta"},
            ]
        },
    )
    env1 = RetrievalEnv()
    env2 = RetrievalEnv()

    env1.reset(task)
    env2.reset(task)
    action = Action(tool_name="search_docs", args={"query": "test", "top_k": 2})

    result1 = env1.step(action)
    result2 = env2.step(action)

    assert result1.model_dump() == result2.model_dump()


def test_files_determinism() -> None:
    task = _task("files", "files_v1", {"files": {"/a.txt": "hello"}})
    env1 = FilesEnv()
    env2 = FilesEnv()

    env1.reset(task)
    env2.reset(task)
    action = Action(tool_name="write_file", args={"path": "/b.txt", "content": "world"})

    result1 = env1.step(action)
    result2 = env2.step(action)

    assert result1.model_dump() == result2.model_dump()
    assert env1.snapshot() == env2.snapshot()


def test_scheduler_determinism() -> None:
    task = _task("scheduling", "scheduler_v1", {"jobs": []})
    env1 = SchedulerEnv()
    env2 = SchedulerEnv()

    env1.reset(task)
    env2.reset(task)
    action = Action(tool_name="create_job", args={"name": "backup", "cron": "0 * * * *", "command": "run"})

    result1 = env1.step(action)
    result2 = env2.step(action)

    assert result1.model_dump() == result2.model_dump()
    assert env1.snapshot() == env2.snapshot()
