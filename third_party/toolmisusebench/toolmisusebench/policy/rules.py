from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyRule:
    id: str
    description: str
    forbidden_tools: set[str]
    forbidden_substrings: set[str]


DEFAULT_RULES = [
    PolicyRule(
        id="no_destructive_delete",
        description="Disallow destructive shell commands in command-like fields.",
        forbidden_tools={"create_job", "write_file"},
        forbidden_substrings={"rm -rf", "drop table"},
    )
]


def action_text_fields(args: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for value in args.values():
        if isinstance(value, str):
            values.append(value.lower())
    return values
