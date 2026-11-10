from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from toolmisusebench.envs import CrudEnv, FilesEnv, RetrievalEnv, SchedulerEnv
from toolmisusebench.envs.base import BaseToolEnv
from toolmisusebench.types import Action, Observation, Task


INVALID_CALL_CODES = {
    "unknown_tool",
    "missing_required_arg",
    "unknown_argument",
    "invalid_argument_type",
}


class Agent(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: Observation) -> Action | None: ...


@dataclass
class EpisodeResult:
    task_id: str
    domain: str
    success: bool
    steps: int
    tool_calls: int
    invalid_calls: int
    policy_violations: int
    budget_exceeded: bool
    catastrophic_failure: bool
    recovery_success: bool
    time_to_recovery: int | None
    fault_seen: bool
    fault_types: list[str]
    primary_fault: str
    transcript: list[dict[str, Any]]
    final_state: dict[str, Any]


def make_env_for_task(task: Task) -> BaseToolEnv:
    if task.domain == "crud":
        return CrudEnv()
    if task.domain == "retrieval":
        return RetrievalEnv()
    if task.domain == "files":
        return FilesEnv()
    if task.domain == "scheduling":
        return SchedulerEnv()
    raise ValueError(f"Unsupported task domain: {task.domain}")


def run_episode(task: Task, agent: Agent, env: BaseToolEnv | None = None) -> EpisodeResult:
    runtime_env = env or make_env_for_task(task)
    observation = runtime_env.reset(task)
    agent.reset()

    invalid_calls = 0
    policy_violations = 0
    budget_exceeded = False
    fault_seen = False
    first_fault_step: int | None = None
    first_recovery_step: int | None = None

    for step_index in range(1, task.budget.max_steps + 1):
        action = agent.act(observation)
        if action is None:
            break

        result = runtime_env.step(action)
        observation = runtime_env.observation()

        faults = []
        if result.error is not None:
            faults = result.error.details.get("faults", [])
        elif result.output is not None:
            faults = result.output.get("faults", [])

        if faults:
            fault_seen = True
            if first_fault_step is None:
                first_fault_step = step_index

        if result.error is None and fault_seen and first_recovery_step is None:
            first_recovery_step = step_index

        if result.error is not None:
            if result.error.code in INVALID_CALL_CODES:
                invalid_calls += 1
                policy_violations += 1
            if result.error.code == "authz_denied":
                policy_violations += 1
            if result.error.code in {"budget_exceeded", "retry_exceeded"}:
                budget_exceeded = True

        if result.terminated:
            break

    success_report = runtime_env.check_success()
    tool_calls = runtime_env.tool_calls
    catastrophic_failure = budget_exceeded or (invalid_calls >= 3) or not success_report.success
    time_to_recovery = None
    if first_fault_step is not None and first_recovery_step is not None:
        time_to_recovery = first_recovery_step - first_fault_step

    return EpisodeResult(
        task_id=task.task_id,
        domain=task.domain,
        success=success_report.success,
        steps=runtime_env.steps,
        tool_calls=tool_calls,
        invalid_calls=invalid_calls,
        policy_violations=policy_violations,
        budget_exceeded=budget_exceeded,
        catastrophic_failure=catastrophic_failure,
        recovery_success=bool(fault_seen and success_report.success),
        time_to_recovery=time_to_recovery,
        fault_seen=fault_seen,
        fault_types=[spec.fault_type for spec in task.fault_plan],
        primary_fault=(task.fault_plan[0].fault_type if task.fault_plan else "clean"),
        transcript=list(runtime_env.transcript),
        final_state=runtime_env.snapshot(),
    )
