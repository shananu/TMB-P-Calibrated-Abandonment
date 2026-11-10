from __future__ import annotations

from copy import deepcopy
from typing import Any

from toolmisusebench.envs.base import BaseToolEnv
from toolmisusebench.types import Action, ErrorInfo, StepResult


class CrudEnv(BaseToolEnv):
    TOOLSET_ID = "crud_v1"
    TOOL_SCHEMAS = [
        {
            "name": "get_account",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
        {
            "name": "update_account",
            "properties": {"id": {"type": "string"}, "patch": {"type": "object"}},
            "required": ["id", "patch"],
            "additionalProperties": False,
        },
        {
            "name": "create_account",
            "properties": {"fields": {"type": "object"}},
            "required": ["fields"],
            "additionalProperties": False,
        },
        {
            "name": "close_account",
            "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["id", "reason"],
            "additionalProperties": False,
        },
    ]

    def _execute_action(self, action: Action) -> StepResult:
        accounts: dict[str, dict[str, Any]] = self.state.setdefault("accounts", {})

        if action.tool_name == "get_account":
            account_id = action.args["id"]
            account = accounts.get(account_id)
            if account is None:
                return StepResult(error=ErrorInfo(code="not_found", message="Account not found."))
            return StepResult(output={"account": deepcopy(account)})

        if action.tool_name == "update_account":
            account_id = action.args["id"]
            patch = action.args["patch"]
            if account_id not in accounts:
                return StepResult(error=ErrorInfo(code="not_found", message="Account not found."))
            accounts[account_id].update(deepcopy(patch))
            return StepResult(output={"account": deepcopy(accounts[account_id])})

        if action.tool_name == "create_account":
            fields = deepcopy(action.args["fields"])
            account_id = fields.get("id") or f"acct_{len(accounts) + 1}"
            if account_id in accounts:
                return StepResult(error=ErrorInfo(code="already_exists", message="Account already exists."))
            fields["id"] = account_id
            fields.setdefault("status", "active")
            accounts[account_id] = fields
            return StepResult(output={"account": deepcopy(fields)})

        if action.tool_name == "close_account":
            account_id = action.args["id"]
            reason = action.args["reason"]
            account = accounts.get(account_id)
            if account is None:
                return StepResult(error=ErrorInfo(code="not_found", message="Account not found."))
            account["status"] = "closed"
            account["closed_reason"] = reason
            return StepResult(output={"account": deepcopy(account)})

        return StepResult(error=ErrorInfo(code="unknown_tool", message="Unknown tool."))
