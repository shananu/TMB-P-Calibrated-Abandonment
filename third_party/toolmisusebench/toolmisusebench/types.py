from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Difficulty = Literal["easy", "medium", "hard"]
Domain = Literal["crud", "retrieval", "files", "scheduling", "mixed"]
Split = Literal["train", "dev", "test_public"]


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(ge=1)
    max_tool_calls: int = Field(ge=1)
    max_retries: int = Field(ge=0)
    timeout_ms: int = Field(ge=1)


class FaultSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_type: Literal["timeout", "rate_limit", "schema_drift", "authz", "adversarial_error"]
    trigger: dict[str, Any]
    severity: Literal["soft", "hard"] = "soft"
    payload: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    split: Split
    difficulty: Difficulty
    domain: Domain
    instruction: str
    toolset_id: str
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[dict[str, Any]] = Field(default_factory=list)
    budget: Budget
    fault_plan: list[FaultSpec] = Field(default_factory=list)
    gold_summary: str | None = None
    seed: int = 0


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RemainingBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps_left: int
    tool_calls_left: int
    retries_left: int


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    tool_schemas: list[dict[str, Any]]
    transcript: list[dict[str, Any]]
    remaining_budget: RemainingBudget
    last_error: ErrorInfo | None = None


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    terminated: bool = False


class SuccessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
