from __future__ import annotations

from copy import deepcopy

from toolmisusebench.faults.base import BaseFaultInjector, FaultEvent
from toolmisusebench.types import Action


class SchemaDriftInjector(BaseFaultInjector):
    def apply(self, action: Action, call_index: int) -> tuple[Action, FaultEvent | None]:
        if not self.should_trigger(action, call_index):
            return action, None

        payload = self.spec.payload
        mutated_args = deepcopy(action.args)

        rename_args = payload.get("rename_args", {})
        if isinstance(rename_args, dict):
            for old_name, new_name in rename_args.items():
                if old_name in mutated_args:
                    mutated_args[new_name] = mutated_args.pop(old_name)

        type_changes = payload.get("type_changes", {})
        if isinstance(type_changes, dict):
            for arg_name, target_type in type_changes.items():
                if arg_name not in mutated_args:
                    continue
                if target_type == "string":
                    mutated_args[arg_name] = str(mutated_args[arg_name])
                elif target_type == "integer":
                    try:
                        mutated_args[arg_name] = int(mutated_args[arg_name])
                    except (TypeError, ValueError):
                        pass

        add_required = payload.get("add_required", {})
        if isinstance(add_required, dict):
            for arg_name, default in add_required.items():
                if arg_name not in mutated_args:
                    mutated_args[arg_name] = default

        return Action(tool_name=action.tool_name, args=mutated_args), self.event()
