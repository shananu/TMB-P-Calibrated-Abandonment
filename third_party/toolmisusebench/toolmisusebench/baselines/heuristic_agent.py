from __future__ import annotations

from typing import Any

from toolmisusebench.baselines.common import default_for_schema
from toolmisusebench.types import Action, Observation


class HeuristicAgent:
    """Simple deterministic baseline agent."""

    def __init__(self) -> None:
        self._attempted_tools: set[str] = set()

    def reset(self) -> None:
        self._attempted_tools = set()

    def act(self, observation: Observation) -> Action | None:
        for tool in observation.tool_schemas:
            tool_name = tool.get("name")
            if not isinstance(tool_name, str):
                continue
            if tool_name in self._attempted_tools:
                continue

            properties = tool.get("properties", {})
            required = tool.get("required", [])
            args: dict[str, Any] = {}
            for field in required:
                schema = properties.get(field, {})
                args[field] = default_for_schema(schema)

            self._attempted_tools.add(tool_name)
            return Action(tool_name=tool_name, args=args)

        return None
