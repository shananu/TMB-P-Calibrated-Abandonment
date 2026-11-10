from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

from toolmisusebench.faults.base import FaultEngine
from toolmisusebench.registry import ToolRegistry
from toolmisusebench.types import Action, ErrorInfo, Observation, RemainingBudget, StepResult, SuccessReport, Task
from toolmisusebench.utils.seed import make_rng


class ToolEnv(Protocol):
    def reset(self, task: Task) -> Observation: ...

    def step(self, action: Action) -> StepResult: ...

    def check_success(self) -> SuccessReport: ...

    def snapshot(self) -> dict[str, Any]: ...


def _path_get(root: Any, path: str) -> Any:
    current = root
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                raise KeyError(path)
            current = current[segment]
        elif isinstance(current, list):
            idx = int(segment)
            current = current[idx]
        else:
            raise KeyError(path)
    return current


def _path_get_maybe(root: Any, path: str) -> tuple[bool, Any]:
    try:
        return True, _path_get(root, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return False, None


class BaseToolEnv:
    TOOLSET_ID: str = ""
    TOOL_SCHEMAS: list[dict[str, Any]] = []

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()
        if self.TOOLSET_ID:
            self.registry.register_toolset(self.TOOLSET_ID, self.TOOL_SCHEMAS)

        self.task: Task | None = None
        self.state: dict[str, Any] = {}
        self.transcript: list[dict[str, Any]] = []
        self.last_error: ErrorInfo | None = None
        self.steps = 0
        self.tool_calls = 0
        self.retries = 0
        self.rng = make_rng(0)
        self.fault_engine: FaultEngine | None = None

    def reset(self, task: Task) -> Observation:
        self.task = task
        self.state = deepcopy(task.initial_state)
        self.transcript = []
        self.last_error = None
        self.steps = 0
        self.tool_calls = 0
        self.retries = 0
        self.rng = make_rng(task.seed)
        self.fault_engine = FaultEngine.from_fault_plan(task.fault_plan, task.seed)
        return self._observation()

    def step(self, action: Action) -> StepResult:
        if self.task is None:
            err = ErrorInfo(code="not_initialized", message="Environment must be reset before step().")
            self.last_error = err
            return StepResult(error=err, terminated=True)

        self.steps += 1
        self.tool_calls += 1

        budget = self.task.budget
        if self.steps > budget.max_steps or self.tool_calls > budget.max_tool_calls:
            err = ErrorInfo(code="budget_exceeded", message="Episode budget exceeded.")
            self.last_error = err
            self._record(action, None, err)
            return StepResult(error=err, terminated=True)

        effective_action = action
        fault_events: list[dict[str, Any]] = []
        if self.fault_engine is not None:
            effective_action, injected_error, events = self.fault_engine.before_call(action, self.tool_calls)
            if events:
                fault_events.extend(event.__dict__ for event in events)
            if injected_error is not None:
                transformed = injected_error
                transformed_events: list[dict[str, Any]] = []
                transformed, extra_events = self.fault_engine.transform_error(effective_action, self.tool_calls, transformed)
                if extra_events:
                    transformed_events.extend(event.__dict__ for event in extra_events)
                if transformed_events:
                    transformed.details.setdefault("faults", [])
                    transformed.details["faults"].extend(transformed_events)
                return self._handle_error(effective_action, transformed)

        schema_err = self.registry.validate_action(self.TOOLSET_ID, effective_action)
        if schema_err is not None:
            if fault_events:
                schema_err.details.setdefault("faults", [])
                schema_err.details["faults"].extend(fault_events)
            return self._handle_error(effective_action, schema_err)

        result = self._execute_action(effective_action)
        if result.error is not None and self.fault_engine is not None:
            transformed_error, events = self.fault_engine.transform_error(effective_action, self.tool_calls, result.error)
            if events:
                transformed_error.details.setdefault("faults", [])
                transformed_error.details["faults"].extend(event.__dict__ for event in events)
            if fault_events:
                transformed_error.details.setdefault("faults", [])
                transformed_error.details["faults"].extend(fault_events)
            result.error = transformed_error

        if result.error is not None:
            return self._handle_error(effective_action, result.error)

        if result.error is None and fault_events:
            result.output = result.output or {}
            result.output.setdefault("faults", [])
            result.output["faults"].extend(fault_events)

        self.last_error = result.error
        self._record(effective_action, result.output, result.error)
        return result

    def _handle_error(self, action: Action, error: ErrorInfo) -> StepResult:
        if self.task is None:
            self.last_error = error
            self._record(action, None, error)
            return StepResult(error=error, terminated=True)

        self.retries += 1
        max_retries = self.task.budget.max_retries
        if self.retries > max_retries:
            capped = ErrorInfo(
                code="retry_exceeded",
                message="Retry budget exceeded.",
                details={"max_retries": max_retries, "original_error_code": error.code},
            )
            self.last_error = capped
            self._record(action, None, capped)
            return StepResult(error=capped, terminated=True)

        self.last_error = error
        self._record(action, None, error)
        return StepResult(error=error, terminated=False)

    def check_success(self) -> SuccessReport:
        if self.task is None:
            return SuccessReport(success=False, details={"reason": "not_initialized"})

        checks: dict[str, bool] = {}
        for idx, criterion in enumerate(self.task.success_criteria):
            op = criterion.get("op")
            key = f"check_{idx}:{op}"
            checks[key] = self._evaluate_criterion(criterion)
        success = all(checks.values()) if checks else False
        return SuccessReport(success=success, checks=checks)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def observation(self) -> Observation:
        return self._observation()

    def _evaluate_criterion(self, criterion: dict[str, Any]) -> bool:
        try:
            op = criterion["op"]
            if op == "successful_tool_call":
                tool_name = criterion["tool_name"]
                min_count = int(criterion.get("min_count", 1))
                count = sum(
                    1
                    for entry in self.transcript
                    if entry.get("tool_name") == tool_name and entry.get("error") is None
                )
                return count >= min_count
            if op == "tool_call":
                tool_name = criterion["tool_name"]
                min_count = int(criterion.get("min_count", 1))
                count = sum(1 for entry in self.transcript if entry.get("tool_name") == tool_name)
                return count >= min_count
            if op == "tool_output_path_equals":
                tool_name = criterion["tool_name"]
                output_path = criterion["output_path"]
                expected = criterion.get("value")
                for entry in reversed(self.transcript):
                    if entry.get("tool_name") != tool_name or entry.get("error") is not None:
                        continue
                    exists, value = _path_get_maybe(entry.get("output"), output_path)
                    if exists:
                        return value == expected
                return False
            if op == "tool_output_path_exists":
                tool_name = criterion["tool_name"]
                output_path = criterion["output_path"]
                for entry in reversed(self.transcript):
                    if entry.get("tool_name") != tool_name or entry.get("error") is not None:
                        continue
                    exists, value = _path_get_maybe(entry.get("output"), output_path)
                    if exists and value is not None:
                        return True
                return False

            path = criterion["path"]
            value = _path_get(self.state, path)
            if op == "equals":
                return value == criterion.get("value")
            if op == "exists":
                return value is not None
            if op == "in":
                container = value
                return criterion.get("value") in container
            if op == "map_value_equals":
                map_key = criterion["key"]
                expected = criterion.get("value")
                if not isinstance(value, dict):
                    return False
                return value.get(map_key) == expected
        except (KeyError, IndexError, TypeError, ValueError):
            return False
        return False

    def _record(self, action: Action, output: dict[str, Any] | None, error: ErrorInfo | None) -> None:
        self.transcript.append(
            {
                "tool_name": action.tool_name,
                "args": deepcopy(action.args),
                "output": deepcopy(output),
                "error": error.model_dump() if error else None,
            }
        )

    def _observation(self) -> Observation:
        if self.task is None:
            raise RuntimeError("Environment not initialized")
        budget = self.task.budget
        remaining = RemainingBudget(
            steps_left=max(0, budget.max_steps - self.steps),
            tool_calls_left=max(0, budget.max_tool_calls - self.tool_calls),
            retries_left=max(0, budget.max_retries - self.retries),
        )
        return Observation(
            instruction=self.task.instruction,
            tool_schemas=self.registry.get_toolset(self.TOOLSET_ID),
            transcript=deepcopy(self.transcript),
            remaining_budget=remaining,
            last_error=self.last_error,
        )

    def _execute_action(self, action: Action) -> StepResult:
        raise NotImplementedError
