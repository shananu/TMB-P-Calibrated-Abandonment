from __future__ import annotations

from copy import deepcopy
from typing import Any

from toolmisusebench.envs.base import BaseToolEnv
from toolmisusebench.types import Action, ErrorInfo, StepResult


class RetrievalEnv(BaseToolEnv):
    TOOLSET_ID = "retrieval_v1"
    TOOL_SCHEMAS = [
        {
            "name": "search_docs",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
        {
            "name": "get_doc",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
            "additionalProperties": False,
        },
    ]

    def _execute_action(self, action: Action) -> StepResult:
        docs: list[dict[str, Any]] = self.state.setdefault("docs", [])

        if action.tool_name == "search_docs":
            query = action.args["query"].strip().lower()
            top_k = max(1, action.args["top_k"])
            query_terms = [term for term in query.split() if term]

            scored: list[tuple[int, str, dict[str, Any]]] = []
            for doc in docs:
                text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
                score = sum(text.count(term) for term in query_terms)
                scored.append((score, doc.get("id", ""), doc))

            scored.sort(key=lambda row: (-row[0], row[1]))
            results = [
                {
                    "id": entry[2].get("id"),
                    "title": entry[2].get("title"),
                    "score": entry[0],
                }
                for entry in scored[:top_k]
                if entry[0] > 0
            ]
            return StepResult(output={"results": results})

        if action.tool_name == "get_doc":
            doc_id = action.args["doc_id"]
            for doc in docs:
                if doc.get("id") == doc_id:
                    return StepResult(output={"doc": deepcopy(doc)})
            return StepResult(error=ErrorInfo(code="not_found", message="Document not found."))

        return StepResult(error=ErrorInfo(code="unknown_tool", message="Unknown tool."))
