# ToolMisuseBench

ToolMisuseBench is an offline, deterministic benchmark for evaluating tool-using agents under tool misuse, execution failures, and recovery constraints.

The project is designed to produce three outputs from one implementation pipeline:

- benchmark package and CLI
- reproducible experiment scripts for paper tables and figure data
- dataset artifacts ready for external publication

## What It Measures

- tool misuse robustness (schema/type/argument errors)
- recovery behavior after injected failures (timeouts, rate limits, authz, drift)
- policy violation behavior under constrained tool interfaces
- budgeted performance under step/tool-call/retry limits

## Features

- deterministic simulator environments: CRUD, retrieval, files, scheduling
- declarative fault injection: schema drift, rate limit, timeout, authz, adversarial errors
- reproducible scoring: task metrics, aggregate metrics, budgeted success curve and AUC
- baseline agents: `heuristic`, `schema_repair`, `policy_aware`, optional `local_llm` adapter
- experiment pipeline for generating report JSON, table CSV/Markdown, and figure CSV artifacts

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart

Generate a deterministic dataset:

```bash
toolmisusebench generate --version v0.1 --out data/toolmisusebench_v0_1 --seed 42
```

Generate a large public-release dataset profile:

```bash
toolmisusebench generate \
  --version v0.1 \
  --out data/toolmisusebench_v0_1 \
  --seed 42 \
  --size-profile large
```

Generate a custom-sized dataset:

```bash
toolmisusebench generate \
  --version v0.1 \
  --out data/toolmisusebench_v0_1 \
  --seed 42 \
  --train-size 5000 \
  --dev-size 800 \
  --test-size 1000
```

Audit generated dataset quality/diversity:

```bash
python -m generator.quality_report --dataset data/toolmisusebench_v0_1 --splits train,dev,test_public
```

Run an evaluation:

```bash
toolmisusebench eval \
  --dataset data/toolmisusebench_v0_1 \
  --split test_public \
  --agent heuristic \
  --report out/report.json
```

Reproduce experiment artifacts:

```bash
toolmisusebench reproduce-paper --config experiments/configs/v0_1.yaml
```

Use a custom agent from your own module:

```bash
toolmisusebench eval \
  --dataset data/toolmisusebench_v0_1 \
  --split test_public \
  --agent-module my_agent_pkg.my_agent:MyAgent \
  --agent-kwargs '{"model_path": "./models/agent.bin"}' \
  --report out/custom_agent_report.json
```

Custom agents must implement:

```python
class MyAgent:
    def reset(self) -> None:
        ...

    def act(self, observation):
        # return Action(tool_name="...", args={...}) or None
        ...
```

CLI help:

```bash
toolmisusebench --help
```

## Outputs

- dataset generation: `data/toolmisusebench_v0_1/{train,dev,test_public}` plus `manifest.json` and version freeze metadata
- evaluation: `out/report.json` and `out/report.traces.jsonl`
- experiment pipeline: `out/experiments/v0_1/` with `results.json`, table files, and figure data CSVs
- optional dataset quality report JSON via `generator.quality_report`

## Repository Layout

- `toolmisusebench/`: core package (types, envs, faults, harness, baselines, CLI)
- `generator/`: generation and export entry scripts
- `experiments/`: experiment runner and artifact builders
- `docs/`: benchmark, scoring, and dataset specifications
- `tests/`: unit and integration coverage

## Reproducibility

- benchmark behavior is deterministic under fixed seeds
- fault triggers are replayable and task-defined
- generated artifacts include checksums in manifest files
- dataset generation runs per-task coherence checks before writing artifacts

## Dataset sizing

- `--size-profile default`: train=24, dev=12, test_public=12
- `--size-profile small`: train=200, dev=50, test_public=50
- `--size-profile medium`: train=1000, dev=200, test_public=300
- `--size-profile large`: train=5000, dev=800, test_public=1000
- custom split sizes are supported via `--train-size`, `--dev-size`, and `--test-size`

## Development

Run tests locally:

```bash
pytest -q
```

CI executes the same test suite on push and pull requests.

## License

MIT. See `LICENSE`.
