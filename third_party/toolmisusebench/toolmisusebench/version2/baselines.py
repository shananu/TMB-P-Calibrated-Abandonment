from __future__ import annotations

import re
from typing import Any

from toolmisusebench.baselines.schema_repair_agent import SchemaRepairAgent
from toolmisusebench.types import Action, Observation


class InstructionParserAgent(SchemaRepairAgent):
    """Deterministic v2 baseline that parses obvious arguments from instructions."""

    def act(self, observation: Observation) -> Action | None:
        if observation.transcript and observation.last_error is None:
            last = observation.transcript[-1]
            if last.get("error") is None:
                return None
        return super().act(observation)

    def _fresh_action(self, observation: Observation) -> Action | None:
        instruction = observation.instruction
        tool_names = {str(tool.get("name")) for tool in observation.tool_schemas}

        action = self._parse_crud(instruction, tool_names)
        if action is not None:
            return action
        action = self._parse_retrieval(instruction, tool_names)
        if action is not None:
            return action
        action = self._parse_files(instruction, tool_names)
        if action is not None:
            return action
        action = self._parse_scheduler(instruction, tool_names)
        if action is not None:
            return action
        return super()._fresh_action(observation)

    def _parse_crud(self, instruction: str, tool_names: set[str]) -> Action | None:
        lowered = instruction.lower()
        account_id = _first_match(r"\b([abn][a-z]?\d{4,6})\b", instruction)

        if account_id and "close_account" in tool_names and any(word in lowered for word in ("close", "deactivate", "archive")):
            reason = _first_match(r"reason(?: as)? ([a-z_]+)", lowered) or "user_request"
            return Action(tool_name="close_account", args={"id": account_id, "reason": reason.rstrip(".")})

        if account_id and "update_account" in tool_names and any(word in lowered for word in ("update", "tier", "patch", "subscription")):
            tier = _first_match(r"(?:tier is|tier=|to) ([a-z_]+)", lowered) or "basic"
            return Action(tool_name="update_account", args={"id": account_id, "patch": {"tier": tier.rstrip(".")}})

        if "create_account" in tool_names and any(word in lowered for word in ("create", "register", "add customer")):
            new_id = account_id or _first_match(r"id ([a-z]\w+)", instruction) or "acct_new"
            return Action(tool_name="create_account", args={"fields": {"id": new_id, "status": "active"}})

        return None
    def _parse_retrieval(self, instruction: str, tool_names: set[str]) -> Action | None:
        lowered = instruction.lower()
        doc_id = _first_match(r"\b(d[a-z]?\d{4,6})\b", instruction)
        if doc_id and "get_doc" in tool_names and any(word in lowered for word in ("document", "doc id", "retrieval record")):
            return Action(tool_name="get_doc", args={"doc_id": doc_id})

        query = _quoted(instruction)
        if query and "search_docs" in tool_names and any(word in lowered for word in ("search", "find relevant", "run document search")):
            return Action(tool_name="search_docs", args={"query": query, "top_k": 1})

        return None

    def _parse_files(self, instruction: str, tool_names: set[str]) -> Action | None:
        lowered = instruction.lower()
        path = _first_match(r"(/[A-Za-z0-9_./-]+)", instruction)
        if path is None:
            return None

        if "write_file" in tool_names and any(word in lowered for word in ("write", "create file", "store the text")):
            content = _quoted(instruction) or ""
            return Action(tool_name="write_file", args={"path": path.rstrip("."), "content": content})

        if "read_file" in tool_names and any(word in lowered for word in ("read", "open", "fetch content")):
            return Action(tool_name="read_file", args={"path": path.rstrip(".")})

        return None

    def _parse_scheduler(self, instruction: str, tool_names: set[str]) -> Action | None:
        lowered = instruction.lower()
        cron = _quoted(instruction) or _first_match(r"cron=([^;]+)", instruction)

        if cron and "validate_cron" in tool_names and any(word in lowered for word in ("validate", "check whether", "cron validation")):
            return Action(tool_name="validate_cron", args={"cron": cron.strip().rstrip(".")})

        if "create_job" in tool_names and any(word in lowered for word in ("create scheduler", "register job", "add a cron job")):
            name = _first_match(r"(?:job|named) ([A-Za-z0-9_]+)", instruction) or "job"
            command = _first_match(r"command '?([A-Za-z0-9_]+)'?", instruction)
            if command is None:
                command = _first_match(r"run ([A-Za-z0-9_]+)", instruction) or "run_task"
            return Action(
                tool_name="create_job",
                args={"name": name.rstrip(";"), "cron": (cron or "0 * * * *").strip(), "command": command.rstrip(".")},
            )

        return None


class InstructionParserNoRecoveryAgent(InstructionParserAgent):
    """Instruction parser ablation that stops rather than repairing after an error."""

    def act(self, observation: Observation) -> Action | None:
        if observation.transcript:
            return None
        return self._fresh_action(observation)


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1)


def _quoted(text: str) -> str | None:
    match = re.search(r"'([^']+)'", text)
    if match is None:
        return None
    return match.group(1)
