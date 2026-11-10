from __future__ import annotations

from toolmisusebench.baselines.local_llm_adapter import LocalLLMAdapter
from toolmisusebench.types import Observation, RemainingBudget


class StubClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, prompt: str) -> str:
        return self.response


def _observation() -> Observation:
    return Observation(
        instruction="Fetch a document",
        tool_schemas=[
            {
                "name": "get_doc",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
                "additionalProperties": False,
            }
        ],
        transcript=[],
        remaining_budget=RemainingBudget(steps_left=5, tool_calls_left=5, retries_left=1),
        last_error=None,
    )


def test_local_llm_adapter_parses_action() -> None:
    adapter = LocalLLMAdapter(StubClient('{"tool_name": "get_doc", "args": {"doc_id": "d1"}}'))
    action = adapter.act(_observation())
    assert action is not None
    assert action.tool_name == "get_doc"
    assert action.args["doc_id"] == "d1"


def test_local_llm_adapter_falls_back_to_heuristic_on_bad_json() -> None:
    adapter = LocalLLMAdapter(StubClient("not json"))
    action = adapter.act(_observation())
    assert action is not None
    assert action.tool_name == "get_doc"
