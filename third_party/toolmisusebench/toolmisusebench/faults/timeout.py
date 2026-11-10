from __future__ import annotations

from toolmisusebench.faults.base import BaseFaultInjector, FaultEvent
from toolmisusebench.types import Action, ErrorInfo


class TimeoutInjector(BaseFaultInjector):
    def maybe_fail(self, action: Action, call_index: int) -> tuple[ErrorInfo | None, FaultEvent | None]:
        if not self.should_trigger(action, call_index):
            return None, None

        err = ErrorInfo(
            code="timeout",
            message="Tool call timed out.",
            details={"timeout_ms": int(self.spec.payload.get("timeout_ms", 1000))},
        )
        return err, self.event()
