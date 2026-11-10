from __future__ import annotations

from copy import deepcopy
from typing import Any

from toolmisusebench.baselines.common import alias_for_field, cast_value, default_for_schema, get_tool_schema
from toolmisusebench.types import Action, Observation


class SchemaRepairAgent:
    """Baseline that retries failed calls with schema-based corrections."""

    def __init__(self) -> None:
        self._attempted_tools: set[str] = set()

    def reset(self) -> None:
        self._attempted_tools = set()

    def act(self, observation: Observation) -> Action | None:
        repaired = self._repair_from_last_error(observation)
        if repaired is not None:
            return repaired

        return self._fresh_action(observation)

    def _fresh_action(self, observation: Observation) -> Action | None:
        for tool in observation.tool_schemas:
            tool_name = tool.get("name")
            if not isinstance(tool_name, str) or tool_name in self._attempted_tools:
                continue

            properties = tool.get("properties", {})
            required = tool.get("required", [])
            args: dict[str, Any] = {}
            for field in required:
                args[field] = default_for_schema(properties.get(field, {}))

            self._attempted_tools.add(tool_name)
            return Action(tool_name=tool_name, args=args)

        return None

    def _repair_from_last_error(self, observation: Observation) -> Action | None:
        error = observation.last_error
        if error is None or not observation.transcript:
            return None

        last_entry = observation.transcript[-1]
        last_tool = last_entry.get("tool_name")
        if not isinstance(last_tool, str):
            return None

        schema = get_tool_schema(observation.tool_schemas, last_tool)
        if schema is None:
            return None

        args = deepcopy(last_entry.get("args", {}))
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if error.code == "unknown_argument":
            unknown = error.details.get("unknown", [])
            for field in unknown:
                args.pop(field, None)

        if error.code == "missing_required_arg":
            missing = error.details.get("missing", [])
            for field in missing:
                if field in args:
                    continue
                alias = alias_for_field(field)
                if alias is not None and alias in args:
                    args[field] = args.pop(alias)
                    continue
                args[field] = default_for_schema(properties.get(field, {}))

        if error.code == "invalid_argument_type":
            arg_name = error.details.get("arg")
            expected = error.details.get("expected")
            if isinstance(arg_name, str) and isinstance(expected, str) and arg_name in args:
                try:
                    args[arg_name] = cast_value(args[arg_name], expected)
                except (TypeError, ValueError):
                    args[arg_name] = default_for_schema(properties.get(arg_name, {}))

        for field in required:
            if field not in args:
                args[field] = default_for_schema(properties.get(field, {}))

        return Action(tool_name=last_tool, args=args)
