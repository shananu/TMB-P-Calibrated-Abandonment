from __future__ import annotations

from copy import deepcopy

from toolmisusebench.envs.base import BaseToolEnv
from toolmisusebench.types import Action, ErrorInfo, StepResult


def _is_valid_field(token: str) -> bool:
    if token == "*":
        return True
    if token.isdigit():
        return 0 <= int(token) <= 59
    return False


def _is_valid_cron(cron: str) -> bool:
    parts = cron.strip().split()
    if len(parts) != 5:
        return False
    return all(_is_valid_field(token) for token in parts)


class SchedulerEnv(BaseToolEnv):
    TOOLSET_ID = "scheduler_v1"
    TOOL_SCHEMAS = [
        {
            "name": "create_job",
            "properties": {
                "name": {"type": "string"},
                "cron": {"type": "string"},
                "command": {"type": "string"},
            },
            "required": ["name", "cron", "command"],
            "additionalProperties": False,
        },
        {
            "name": "validate_cron",
            "properties": {"cron": {"type": "string"}},
            "required": ["cron"],
            "additionalProperties": False,
        },
        {
            "name": "list_jobs",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    ]

    def _execute_action(self, action: Action) -> StepResult:
        jobs: list[dict[str, str]] = self.state.setdefault("jobs", [])

        if action.tool_name == "validate_cron":
            cron = action.args["cron"]
            return StepResult(output={"cron": cron, "valid": _is_valid_cron(cron)})

        if action.tool_name == "create_job":
            name = action.args["name"]
            cron = action.args["cron"]
            command = action.args["command"]
            if not _is_valid_cron(cron):
                return StepResult(error=ErrorInfo(code="invalid_cron", message="Cron expression is invalid."))
            if any(job["name"] == name for job in jobs):
                return StepResult(error=ErrorInfo(code="already_exists", message="Job already exists."))
            job = {"name": name, "cron": cron, "command": command}
            jobs.append(job)
            jobs.sort(key=lambda item: item["name"])
            return StepResult(output={"job": deepcopy(job)})

        if action.tool_name == "list_jobs":
            return StepResult(output={"jobs": deepcopy(jobs)})

        return StepResult(error=ErrorInfo(code="unknown_tool", message="Unknown tool."))
