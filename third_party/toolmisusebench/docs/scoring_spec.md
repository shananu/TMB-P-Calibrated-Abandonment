# Scoring Spec

This document defines per-task and aggregate metrics.

## Per-task metrics

- `TaskSuccess` (0/1)
- `PolicyViolations` (count)
- `InvalidCallRate` (`invalid_calls / tool_calls`)
- `RecoverySuccess` (0/1)
- `TimeToRecovery` (integer steps or null)
- `ToolCallsUsed` (count)
- `BudgetExceeded` (0/1)
- `CatastrophicFailure` (0/1)
- `PrimaryFault` (string)

`BudgetExceeded` is set when `budget_exceeded` or `retry_exceeded` terminates the episode.

## Aggregate metrics

Per-agent aggregate is the mean over tasks for all metrics except `TimeToRecovery`, which averages only non-null values.

## Budgeted success curve

For `k ∈ {4, 8, 16, 32}`:

- success at budget `k` is the fraction of tasks with `TaskSuccess=1` and `ToolCallsUsed <= k`
- AUC is the normalized trapezoidal area over the success-vs-budget curve

## Experiment artifacts

`reproduce-paper` pipeline writes:

- `results.json` with `by_agent` metrics payloads
- `table1_overall.csv`
- `table2_fault_breakdown.csv`
- `tables.md`
- `figure1_budgeted_success.csv`
- `figure2_time_to_recovery.csv`
