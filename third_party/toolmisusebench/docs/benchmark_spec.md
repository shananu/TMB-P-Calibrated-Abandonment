# Benchmark Spec

This document defines benchmark behavior and deterministic execution contracts.

## Determinism contract

- Every task has a fixed `seed`.
- Fault trigger randomness uses seeded RNG streams.
- Running generation with the same `(version, seed)` produces the same task content.
- Running evaluation on the same task and agent produces identical traces and metrics.

## Budget contract

- Episode ends on `max_steps` or `max_tool_calls` overflow (`budget_exceeded`).
- Error responses consume retry budget; exceeding `max_retries` ends episode (`retry_exceeded`).

## Success criteria operators

Task success criteria are evaluated with the following operators:

- `equals`: value at `path` must match `value`
- `exists`: value at `path` must be present and non-null
- `in`: `value` must exist within container at `path`
- `map_value_equals`: dict at `path` must contain `key` with expected `value`
- `tool_call`: transcript must include at least `min_count` calls to `tool_name`
- `successful_tool_call`: transcript must include at least `min_count` successful calls to `tool_name`

## v0.1 domains

- CRUD tools (`crud_v1`)
- Retrieval tools (`retrieval_v1`)
- Files tools (`files_v1`)
- Scheduler tools (`scheduler_v1`)

## v0.1 fault model

- `schema_drift`
- `rate_limit`
- `timeout`
- `authz`
- `adversarial_error`

Fault plans are declarative inside each task and are replayed by the deterministic fault engine.

## v0.1 freeze artifact

The generator writes `v0_1_freeze.json` with:

- benchmark version
- generation seed
- domain set
- split sizes
- supported fault types

This freeze file is intended to be referenced by experiments and dataset publishing.
