from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Any

from toolmisusebench.types import Action, ErrorInfo, FaultSpec


@dataclass
class FaultEvent:
    fault_type: str
    severity: str
    payload: dict[str, Any]


class BaseFaultInjector:
    def __init__(self, fault_spec: FaultSpec, fault_index: int, task_seed: int) -> None:
        self.spec = fault_spec
        self.index = fault_index
        prob_seed = fault_spec.trigger.get("prob", {}).get("seed") if isinstance(fault_spec.trigger.get("prob"), dict) else None
        base_seed = int(prob_seed) if prob_seed is not None else task_seed
        self.rng = Random((base_seed * 9973) + fault_index)

    @property
    def fault_type(self) -> str:
        return self.spec.fault_type

    def event(self) -> FaultEvent:
        return FaultEvent(
            fault_type=self.spec.fault_type,
            severity=self.spec.severity,
            payload=deepcopy(self.spec.payload),
        )

    def should_trigger(self, action: Action, call_index: int) -> bool:
        trigger = self.spec.trigger

        on_tool = trigger.get("on_tool")
        if on_tool is not None and action.tool_name != on_tool:
            return False

        on_nth_call = trigger.get("on_nth_call")
        if on_nth_call is not None and call_index != int(on_nth_call):
            return False

        arg_matches = trigger.get("when_arg_matches")
        if isinstance(arg_matches, dict):
            for key, expected in arg_matches.items():
                if action.args.get(key) != expected:
                    return False

        prob = trigger.get("prob")
        if prob is not None:
            p = float(prob["p"]) if isinstance(prob, dict) else float(prob)
            if self.rng.random() >= p:
                return False

        return True

    def apply(self, action: Action, call_index: int) -> tuple[Action, FaultEvent | None]:
        return action, None

    def maybe_fail(self, action: Action, call_index: int) -> tuple[ErrorInfo | None, FaultEvent | None]:
        return None, None

    def transform_error(self, action: Action, call_index: int, error: ErrorInfo) -> tuple[ErrorInfo, FaultEvent | None]:
        return error, None


class FaultEngine:
    def __init__(self, injectors: list[BaseFaultInjector]) -> None:
        self.schema_drift = [inj for inj in injectors if inj.fault_type == "schema_drift"]
        self.authz = [inj for inj in injectors if inj.fault_type == "authz"]
        self.rate_limit = [inj for inj in injectors if inj.fault_type == "rate_limit"]
        self.timeout = [inj for inj in injectors if inj.fault_type == "timeout"]
        self.adversarial = [inj for inj in injectors if inj.fault_type == "adversarial_error"]

    @classmethod
    def from_fault_plan(cls, fault_plan: list[FaultSpec], task_seed: int) -> "FaultEngine":
        from toolmisusebench.faults.adversarial_errors import AdversarialErrorInjector
        from toolmisusebench.faults.authz import AuthzInjector
        from toolmisusebench.faults.rate_limit import RateLimitInjector
        from toolmisusebench.faults.schema_drift import SchemaDriftInjector
        from toolmisusebench.faults.timeout import TimeoutInjector

        factory = {
            "schema_drift": SchemaDriftInjector,
            "authz": AuthzInjector,
            "rate_limit": RateLimitInjector,
            "timeout": TimeoutInjector,
            "adversarial_error": AdversarialErrorInjector,
        }

        injectors: list[BaseFaultInjector] = []
        for idx, spec in enumerate(fault_plan):
            injector_cls = factory.get(spec.fault_type)
            if injector_cls is None:
                continue
            injectors.append(injector_cls(spec, idx, task_seed))
        return cls(injectors)

    def before_call(self, action: Action, call_index: int) -> tuple[Action, ErrorInfo | None, list[FaultEvent]]:
        working = Action(tool_name=action.tool_name, args=deepcopy(action.args))
        events: list[FaultEvent] = []

        for injector in self.schema_drift:
            working, event = injector.apply(working, call_index)
            if event is not None:
                events.append(event)

        for injector in (self.authz + self.rate_limit + self.timeout):
            err, event = injector.maybe_fail(working, call_index)
            if event is not None:
                events.append(event)
            if err is not None:
                err.details.setdefault("faults", [event.__dict__ for event in events if event is not None])
                return working, err, events

        return working, None, events

    def transform_error(self, action: Action, call_index: int, error: ErrorInfo) -> tuple[ErrorInfo, list[FaultEvent]]:
        transformed = error
        events: list[FaultEvent] = []

        for injector in self.adversarial:
            transformed, event = injector.transform_error(action, call_index, transformed)
            if event is not None:
                events.append(event)

        if events:
            existing_faults = list(transformed.details.get("faults", []))
            existing_faults.extend(event.__dict__ for event in events)
            transformed.details["faults"] = existing_faults

        return transformed, events
