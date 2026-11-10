from __future__ import annotations

from typing import Any

from toolmisusebench.types import Action, ErrorInfo


_TYPE_MAP: dict[str, tuple[type[Any], ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list, tuple),
}


class ToolRegistry:
    def __init__(self) -> None:
        self._toolsets: dict[str, dict[str, dict[str, Any]]] = {}

    def register_toolset(self, toolset_id: str, tools: list[dict[str, Any]]) -> None:
        indexed: dict[str, dict[str, Any]] = {}
        for schema in tools:
            name = schema.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Tool schema must include a non-empty 'name'")
            indexed[name] = schema
        self._toolsets[toolset_id] = indexed

    def get_toolset(self, toolset_id: str) -> list[dict[str, Any]]:
        return list(self._toolsets.get(toolset_id, {}).values())

    def get_schema(self, toolset_id: str, tool_name: str) -> dict[str, Any] | None:
        return self._toolsets.get(toolset_id, {}).get(tool_name)

    def validate_action(self, toolset_id: str, action: Action) -> ErrorInfo | None:
        schema = self.get_schema(toolset_id, action.tool_name)
        if schema is None:
            return ErrorInfo(
                code="unknown_tool",
                message=f"Tool '{action.tool_name}' is not available in toolset '{toolset_id}'.",
                details={"toolset_id": toolset_id, "tool_name": action.tool_name},
            )

        properties: dict[str, dict[str, Any]] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])
        additional_properties: bool = schema.get("additionalProperties", True)

        missing = [field for field in required if field not in action.args]
        if missing:
            return ErrorInfo(
                code="missing_required_arg",
                message="Action is missing required arguments.",
                details={"missing": missing, "tool_name": action.tool_name},
            )

        if not additional_properties:
            unknown = [name for name in action.args if name not in properties]
            if unknown:
                return ErrorInfo(
                    code="unknown_argument",
                    message="Action contains unknown arguments.",
                    details={"unknown": unknown, "tool_name": action.tool_name},
                )

        for arg_name, arg_value in action.args.items():
            arg_schema = properties.get(arg_name)
            if arg_schema is None:
                continue

            expected_type = arg_schema.get("type")
            if expected_type in _TYPE_MAP and not isinstance(arg_value, _TYPE_MAP[expected_type]):
                return ErrorInfo(
                    code="invalid_argument_type",
                    message="Argument type mismatch.",
                    details={
                        "tool_name": action.tool_name,
                        "arg": arg_name,
                        "expected": expected_type,
                        "received": type(arg_value).__name__,
                    },
                )

        return None
