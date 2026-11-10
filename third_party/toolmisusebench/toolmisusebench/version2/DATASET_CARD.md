# ToolMisuseBench v2 Dataset Card

## Intended Use

ToolMisuseBench v2 is intended for evaluating tool-using agents under deterministic tool misuse, execution failure, and recovery conditions. It is designed for offline research on:

- schema and argument correctness
- recovery after transient tool failures
- hidden interface drift
- authorization denial behavior
- adversarial or ambiguous error feedback
- tool-call efficiency under explicit budgets

## Out-of-Scope Use

ToolMisuseBench v2 should not be treated as:

- a broad language understanding benchmark
- a live API reliability benchmark
- a cybersecurity benchmark
- a real user safety benchmark
- a substitute for production integration testing

The environments are synthetic and intentionally controlled.

## Dataset Design

The v2 generator crosses:

- 4 domains: CRUD, retrieval, files, scheduling
- 6 fault families: clean, timeout, rate limit, schema drift, authorization, adversarial error
- 3 difficulty levels: easy, medium, hard
- 3 budget profiles: tight, standard, relaxed

The default public test split contains one full crossed cycle:

- 216 total tasks
- 54 tasks per domain
- 36 tasks per fault family
- 72 tasks per difficulty level
- 72 tasks per budget profile
- 9 tasks per domain-fault cell

Large profiles are available for final reporting and release-scale experiments.

## Success Criteria

v2 supports v1 state and transcript criteria and adds output-grounded criteria:

- `tool_output_path_equals`
- `tool_output_path_exists`

These checks verify structured tool outputs and reduce reliance on successful-call-only completion.

## Known Limitations

- The task environments are synthetic.
- Tool semantics are intentionally compact.
- The default split is a balanced diagnostic slice, not the final recommended scale for all publication results.
- Current policy behavior is lightweight and should not be interpreted as complete safety reasoning.
- Hidden schema drift is represented through backend argument mutation, not through a full schema-version negotiation protocol.

## Reproducibility

Recommended commands:

```bash
toolmisusebench generate --version v0.2 --out data/toolmisusebench_v0_2 --seed 42
python -m generator.quality_report --dataset data/toolmisusebench_v0_2 --splits train,dev,test_public --v2
toolmisusebench reproduce-paper --config experiments/configs/v0_2.yaml
```

Generated artifacts include manifest checksums and `v0_2_freeze.json`.

