from __future__ import annotations

import json
import re
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from typing import Any
from urllib.parse import urlsplit

from toolmisusebench.types import Action, Observation


class OpenAICompatibleToolAgent:
    """Tool-calling agent for a local OpenAI-compatible inference server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: int = 120,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 128,
        disable_thinking: bool = True,
    ) -> None:
        parsed_url = urlsplit(base_url.rstrip("/"))
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("base_url must be an absolute http(s) URL.")
        self.base_url = base_url.rstrip("/")
        self._scheme = parsed_url.scheme
        self._host = parsed_url.netloc
        self._base_path = parsed_url.path.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self._connection: HTTPConnection | HTTPSConnection | None = None
        self._tool_cache: dict[str, list[dict[str, Any]]] = {}

    def reset(self) -> None:
        return None

    def act(self, observation: Observation) -> Action | None:
        if observation.transcript and observation.last_error is None:
            last = observation.transcript[-1]
            if last.get("error") is None:
                return None

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stream": False,
            "messages": self._messages(observation),
            "tools": self._tools(observation),
            "tool_choice": "required",
        }
        response = self._post("/chat/completions", payload)
        return self._parse_action(response)

    def _messages(self, observation: Observation) -> list[dict[str, str]]:
        transcript = json.dumps(observation.transcript, ensure_ascii=True, separators=(",", ":"))
        thinking_instruction = " /no_think" if self.disable_thinking else ""
        return [
            {
                "role": "system",
                "content": (
                    "Select exactly one available tool. Use only permitted fields. Repair or retry "
                    "a failed prior call when appropriate. Return no prose or reasoning."
                    f"{thinking_instruction}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Instruction: {observation.instruction}\n"
                    f"Transcript: {transcript}\n"
                    f"Budget: {observation.remaining_budget.model_dump()}\n"
                    f"Error: {observation.last_error.model_dump() if observation.last_error else None}\n"
                    "Call one tool now."
                ),
            },
        ]

    def _tools(self, observation: Observation) -> list[dict[str, Any]]:
        cache_key = json.dumps(observation.tool_schemas, ensure_ascii=True, sort_keys=True)
        cached = self._tool_cache.get(cache_key)
        if cached is not None:
            return cached
        tools: list[dict[str, Any]] = []
        for schema in observation.tool_schemas:
            name = schema.get("name")
            if not isinstance(name, str):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"ToolMisuseBench tool {name}.",
                        "parameters": {
                            "type": "object",
                            "properties": schema.get("properties", {}),
                            "required": schema.get("required", []),
                            "additionalProperties": schema.get("additionalProperties", False),
                        },
                    },
                }
            )
        self._tool_cache[cache_key] = tools
        return tools

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        request_path = f"{self._base_path}{path}" or path
        last_error: Exception | None = None
        for _ in range(2):
            try:
                connection = self._connection or self._new_connection()
                connection.request(
                    "POST",
                    request_path,
                    body=body,
                    headers={"Content-Type": "application/json", "Connection": "keep-alive"},
                )
                response = connection.getresponse()
                raw = response.read().decode("utf-8")
                if response.status >= 400:
                    raise RuntimeError(f"LLM server returned HTTP {response.status}: {raw}")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise RuntimeError("LLM server returned an unexpected response payload.")
                self._connection = connection
                return parsed
            except (OSError, HTTPException, json.JSONDecodeError) as exc:
                last_error = exc
                self._close_connection()
        raise RuntimeError(f"Could not reach local LLM server at {self.base_url}: {last_error}")

    def _new_connection(self) -> HTTPConnection | HTTPSConnection:
        connection_cls = HTTPSConnection if self._scheme == "https" else HTTPConnection
        return connection_cls(self._host, timeout=self.timeout_s)

    def _close_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def _parse_action(self, response: dict[str, Any]) -> Action | None:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response has no choices.")
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("LLM response message is invalid.")

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            function = tool_calls[0].get("function", {})
            if isinstance(function, dict):
                return _action_from_parts(function.get("name"), function.get("arguments"))

        content = message.get("content")
        if isinstance(content, str):
            return _action_from_content(content)
        return None


def _action_from_parts(tool_name: Any, raw_args: Any) -> Action | None:
    if not isinstance(tool_name, str):
        return None
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return None
    else:
        args = raw_args
    if not isinstance(args, dict):
        return None
    return Action(tool_name=tool_name, args=args)


def _action_from_content(content: str) -> Action | None:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _action_from_parts(payload.get("tool_name"), payload.get("args"))
