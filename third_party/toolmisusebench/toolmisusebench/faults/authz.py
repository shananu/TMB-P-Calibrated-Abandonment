from __future__ import annotations

from toolmisusebench.faults.base import BaseFaultInjector, FaultEvent
from toolmisusebench.types import Action, ErrorInfo


class AuthzInjector(BaseFaultInjector):
    def maybe_fail(self, action: Action, call_index: int) -> tuple[ErrorInfo | None, FaultEvent | None]:
        if not self.should_trigger(action, call_index):
            return None, None

        payload = self.spec.payload
        deny = bool(payload.get("deny", True))
        required_token = payload.get("required_token")
        token_field = str(payload.get("token_field", "token"))

        if required_token is not None:
            if action.args.get(token_field) != required_token:
                deny = True
            else:
                deny = False

        if not deny:
            return None, None

        err = ErrorInfo(
            code="authz_denied",
            message="Authorization failed for this action.",
            details={"required_role": payload.get("required_role")},
        )
        return err, self.event()
