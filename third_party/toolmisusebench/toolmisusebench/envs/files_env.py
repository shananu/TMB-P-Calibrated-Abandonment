from __future__ import annotations

from copy import deepcopy

from toolmisusebench.envs.base import BaseToolEnv
from toolmisusebench.types import Action, ErrorInfo, StepResult


class FilesEnv(BaseToolEnv):
    TOOLSET_ID = "files_v1"
    TOOL_SCHEMAS = [
        {
            "name": "read_file",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        {
            "name": "write_file",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        {
            "name": "list_dir",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    ]

    def _execute_action(self, action: Action) -> StepResult:
        files: dict[str, str] = self.state.setdefault("files", {})

        if action.tool_name == "read_file":
            path = action.args["path"]
            if path not in files:
                return StepResult(error=ErrorInfo(code="not_found", message="File not found."))
            return StepResult(output={"path": path, "content": files[path]})

        if action.tool_name == "write_file":
            path = action.args["path"]
            content = action.args["content"]
            files[path] = content
            return StepResult(output={"path": path, "bytes_written": len(content)})

        if action.tool_name == "list_dir":
            prefix = action.args["path"].rstrip("/")
            normalized = "" if prefix in ("", "/") else f"{prefix}/"
            entries: set[str] = set()

            for file_path in files:
                if not file_path.startswith(normalized):
                    continue
                remainder = file_path[len(normalized) :]
                if not remainder:
                    continue
                entries.add(remainder.split("/", 1)[0])

            return StepResult(output={"path": action.args["path"], "entries": sorted(entries)})

        return StepResult(error=ErrorInfo(code="unknown_tool", message="Unknown tool."))

    def snapshot(self) -> dict[str, object]:
        return deepcopy(self.state)
