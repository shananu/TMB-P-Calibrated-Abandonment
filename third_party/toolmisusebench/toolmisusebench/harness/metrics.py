from __future__ import annotations

from dataclasses import asdict
from typing import Any

from toolmisusebench.harness.runner import EpisodeResult


def task_metrics(result: EpisodeResult) -> dict[str, Any]:
    invalid_call_rate = 0.0
    if result.tool_calls > 0:
        invalid_call_rate = result.invalid_calls / result.tool_calls

    return {
        "task_id": result.task_id,
        "domain": result.domain,
        "PrimaryFault": result.primary_fault,
        "FaultTypes": list(result.fault_types),
        "TaskSuccess": int(result.success),
        "PolicyViolations": result.policy_violations,
        "InvalidCallRate": invalid_call_rate,
        "RecoverySuccess": int(result.recovery_success),
        "TimeToRecovery": result.time_to_recovery,
        "ToolCallsUsed": result.tool_calls,
        "BudgetExceeded": int(result.budget_exceeded),
        "CatastrophicFailure": int(result.catastrophic_failure),
    }


def budgeted_success_curve(
    task_metrics_rows: list[dict[str, Any]],
    caps: tuple[int, ...] = (4, 8, 16, 32),
) -> dict[str, Any]:
    if not task_metrics_rows:
        return {"caps": list(caps), "success": [0.0 for _ in caps], "auc": 0.0}

    success_rates: list[float] = []
    n_tasks = len(task_metrics_rows)

    for cap in caps:
        cap_success = sum(
            1
            for row in task_metrics_rows
            if row["TaskSuccess"] == 1 and int(row["ToolCallsUsed"]) <= cap
        )
        success_rates.append(cap_success / n_tasks)

    auc = _normalized_trapezoid_auc(list(caps), success_rates)
    return {"caps": list(caps), "success": success_rates, "auc": auc}


def _normalized_trapezoid_auc(xs: list[int], ys: list[float]) -> float:
    if len(xs) < 2:
        return ys[0] if ys else 0.0
    area = 0.0
    for idx in range(1, len(xs)):
        width = xs[idx] - xs[idx - 1]
        height = (ys[idx] + ys[idx - 1]) / 2.0
        area += width * height
    max_area = float(xs[-1] - xs[0])
    if max_area <= 0:
        return 0.0
    return area / max_area


def episode_to_trace_row(result: EpisodeResult) -> dict[str, Any]:
    row = asdict(result)
    return row
