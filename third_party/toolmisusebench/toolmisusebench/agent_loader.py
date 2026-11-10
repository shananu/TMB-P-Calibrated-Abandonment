from __future__ import annotations

import importlib
import json
from typing import Any, Protocol

from toolmisusebench.baselines import build_agent
from toolmisusebench.types import Action, Observation


class RuntimeAgent(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: Observation) -> Action | None: ...


class AgentLoadError(ValueError):
    pass


def build_runtime_agent(
    agent_name: str | None,
    agent_module: str | None,
    agent_kwargs_raw: str,
) -> RuntimeAgent:
    kwargs = _parse_agent_kwargs(agent_kwargs_raw)

    if agent_module is not None:
        return _load_custom_agent(agent_module, kwargs)

    if agent_name is None:
        raise AgentLoadError("Either --agent or --agent-module must be provided.")

    if kwargs:
        raise AgentLoadError("--agent-kwargs is only supported with --agent-module.")

    return build_agent(agent_name)


def _parse_agent_kwargs(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AgentLoadError(f"Invalid --agent-kwargs JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise AgentLoadError("--agent-kwargs must be a JSON object.")
    return parsed


def _load_custom_agent(spec: str, kwargs: dict[str, Any]) -> RuntimeAgent:
    module_name, attr_path = _parse_spec(spec)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise AgentLoadError(f"Could not import module '{module_name}' from --agent-module.") from exc

    target: Any = module
    for part in attr_path.split("."):
        if not hasattr(target, part):
            raise AgentLoadError(f"Attribute '{part}' not found in '{spec}'.")
        target = getattr(target, part)

    if not callable(target):
        raise AgentLoadError(f"Target '{spec}' is not callable.")

    try:
        agent = target(**kwargs)
    except TypeError as exc:
        raise AgentLoadError(f"Could not initialize custom agent '{spec}': {exc}") from exc

    _validate_agent(agent)
    return agent


def _parse_spec(spec: str) -> tuple[str, str]:
    raw = spec.strip()
    if not raw:
        raise AgentLoadError("--agent-module cannot be empty.")
    if ":" in raw:
        module_name, attr_path = raw.split(":", 1)
        if not module_name or not attr_path:
            raise AgentLoadError("--agent-module must use 'module:ClassName' format.")
        return module_name, attr_path
    return raw, "Agent"


def _validate_agent(agent: Any) -> None:
    if not hasattr(agent, "reset") or not callable(getattr(agent, "reset")):
        raise AgentLoadError("Custom agent must implement a callable reset() method.")
    if not hasattr(agent, "act") or not callable(getattr(agent, "act")):
        raise AgentLoadError("Custom agent must implement a callable act(observation) method.")
