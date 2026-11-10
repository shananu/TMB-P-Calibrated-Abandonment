from __future__ import annotations

from toolmisusebench.faults.base import BaseFaultInjector, FaultEvent
from toolmisusebench.types import Action, ErrorInfo


class RateLimitInjector(BaseFaultInjector):
    def maybe_fail(self, action: Action, call_index: int) -> tuple[ErrorInfo | None, FaultEvent | None]:
        if not self.should_trigger(action, call_index):
            return None, None

        after_n_calls = int(self.spec.payload.get("after_n_calls", 0))
        if call_index <= after_n_calls:
            return None, None

        err = ErrorInfo(
            code="rate_limited",
            message="Rate limit exceeded.",
            details={"retry_after_ms": int(self.spec.payload.get("retry_after_ms", 250))},
        )
        return err, self.event()
