from __future__ import annotations

from dataclasses import dataclass

from toolmisusebench.policy.rules import DEFAULT_RULES, action_text_fields
from toolmisusebench.types import Action


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None


def evaluate_action(action: Action) -> PolicyDecision:
    text_values = action_text_fields(action.args)
    for rule in DEFAULT_RULES:
        if action.tool_name not in rule.forbidden_tools:
            continue
        for forbidden in rule.forbidden_substrings:
            if any(forbidden in value for value in text_values):
                return PolicyDecision(allowed=False, reason=rule.id)
    return PolicyDecision(allowed=True)
