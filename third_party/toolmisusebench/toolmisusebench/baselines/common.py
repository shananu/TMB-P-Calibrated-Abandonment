from __future__ import annotations

from typing import Any


def get_tool_schema(tool_schemas: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    for schema in tool_schemas:
        if schema.get("name") == tool_name:
            return schema
    return None


def default_for_schema(arg_schema: dict[str, Any]) -> Any:
    arg_type = arg_schema.get("type")
    if arg_type == "string":
        return "value"
    if arg_type == "integer":
        return 1
    if arg_type == "number":
        return 1.0
    if arg_type == "boolean":
        return False
    if arg_type == "object":
        return {}
    if arg_type == "array":
        return []
    return None


def cast_value(value: Any, target_type: str) -> Any:
    if target_type == "string":
        return str(value)
    if target_type == "integer":
        return int(value)
    if target_type == "number":
        return float(value)
    if target_type == "boolean":
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        return bool(value)
    return value


def alias_for_field(field_name: str) -> str | None:
    aliases = {
        "id": "doc_id",
        "doc_id": "id",
        "account_id": "id",
    }
    return aliases.get(field_name)
