from __future__ import annotations

from copy import deepcopy

from toolmisusebench.faults.base import BaseFaultInjector, FaultEvent
from toolmisusebench.types import Action, ErrorInfo


class AdversarialErrorInjector(BaseFaultInjector):
    def transform_error(self, action: Action, call_index: int, error: ErrorInfo) -> tuple[ErrorInfo, FaultEvent | None]:
        if not self.should_trigger(action, call_index):
            return error, None

        messages = self.spec.payload.get("messages")
        if isinstance(messages, list) and messages:
            message = str(messages[int(self.rng.random() * len(messages)) % len(messages)])
        else:
            message = str(self.spec.payload.get("message", "Unexpected internal failure."))

        transformed = ErrorInfo(
            code=str(self.spec.payload.get("code", error.code)),
            message=message,
            details=deepcopy(error.details),
        )
        transformed.details["original_error_code"] = error.code
        return transformed, self.event()
