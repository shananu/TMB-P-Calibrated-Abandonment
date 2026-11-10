# Dataset Schema

This document defines task records and split artifact conventions.

## Record schema (v0.1)

Each task record is serialized as one JSON object per line (`tasks.jsonl`) and includes:

- `task_id` (string)
- `split` (`train | dev | test_public`)
- `difficulty` (`easy | medium | hard`)
- `domain` (`crud | retrieval | files | scheduling | mixed`)
- `instruction` (string)
- `toolset_id` (string)
- `tool_schemas` (array of JSON-schema-like tool specs)
- `initial_state` (object)
- `success_criteria` (array of criterion objects)
- `budget` object
  - `max_steps` (int)
  - `max_tool_calls` (int)
  - `max_retries` (int)
  - `timeout_ms` (int)
- `fault_plan` (array of fault objects)
- `gold_summary` (string or null)
- `seed` (int)

## Split layout (v0.1)

`toolmisusebench generate --version v0.1 --out <path> --seed <n>` writes:

- `<path>/train/tasks.jsonl`
- `<path>/dev/tasks.jsonl`
- `<path>/test_public/tasks.jsonl`
- optional parquet mirrors when `pyarrow` is available (`tasks.parquet`)
- `<path>/manifest.json` (sha256 + bytes for generated artifacts)
- `<path>/v0_1_freeze.json` (version freeze metadata)

Default deterministic split sizes in v0.1:

- train: 24
- dev: 12
- test_public: 12

Configurable size profiles:

- `default`: train=24, dev=12, test_public=12
- `small`: train=200, dev=50, test_public=50
- `medium`: train=1000, dev=200, test_public=300
- `large`: train=5000, dev=800, test_public=1000

CLI users can also fully override split sizes with:

- `--train-size <n>`
- `--dev-size <n>`
- `--test-size <n>`

When split overrides are provided, all three values are required.
