from __future__ import annotations

from toolmisusebench.baselines.common import default_for_schema
from toolmisusebench.baselines.schema_repair_agent import SchemaRepairAgent
from toolmisusebench.policy.evaluator import evaluate_action
from toolmisusebench.types import Action, Observation


class PolicyAwareAgent(SchemaRepairAgent):
    """Schema-repair agent that blocks unsafe actions before execution."""

    def __init__(self) -> None:
        super().__init__()
        self._blocked_tools: set[str] = set()

    def reset(self) -> None:
        super().reset()
        self._blocked_tools = set()

    def act(self, observation: Observation) -> Action | None:
        if observation.last_error is not None and observation.last_error.code == "authz_denied":
            last_tool = self._last_tool_name(observation)
            if last_tool is not None:
                self._blocked_tools.add(last_tool)

        proposal = super().act(observation)
        if proposal is None:
            return None

        if proposal.tool_name in self._blocked_tools:
            return self._safe_fallback(observation)

        decision = evaluate_action(proposal)
        if decision.allowed:
            return proposal

        self._blocked_tools.add(proposal.tool_name)
        return self._safe_fallback(observation)

    def _last_tool_name(self, observation: Observation) -> str | None:
        if not observation.transcript:
            return None
        value = observation.transcript[-1].get("tool_name")
        if isinstance(value, str):
            return value
        return None

    def _safe_fallback(self, observation: Observation) -> Action | None:
        for tool in observation.tool_schemas:
            tool_name = tool.get("name")
            if not isinstance(tool_name, str):
                continue
            if tool_name in self._blocked_tools:
                continue
            if tool_name.startswith("get_") or tool_name.startswith("list_") or tool_name.startswith("search_"):
                properties = tool.get("properties", {})
                required = tool.get("required", [])
                args = {field: default_for_schema(properties.get(field, {})) for field in required}
                return Action(tool_name=tool_name, args=args)
        return None
