# ToolMisuseBench v2 Roadmap

ToolMisuseBench v2 should preserve the core v1 identity:

- offline execution
- deterministic replay
- explicit tool-call, step, and retry budgets
- declarative fault plans
- dataset-first benchmark release
- transparent scoring and trace artifacts

The goal is not to turn ToolMisuseBench into a broad agent benchmark. The goal is to make the existing tool-misuse and recovery benchmark harder to dismiss, easier to reproduce, and more useful for researchers building reliable tool-using agents.

## Main v2 Improvements

### 1. Deconfounded Dataset Generation

v1 couples domain, difficulty, operation type, and fault type through deterministic index patterns. This makes fault-conditioned results hard to interpret.

v2 should independently sample or explicitly cross:

- domain
- task operation
- fault type
- difficulty
- budget profile
- recovery requirement

Recommended v2 target: every fault family should appear in every compatible domain, across easy, medium, and hard tasks.

### 2. Stronger Success Criteria

v1 often accepts transcript-based criteria such as "successful tool call." This is useful for measuring interaction behavior but can be too weak as a task-completion signal.

v2 should prefer final-state or output-grounded checks:

- retrieval tasks should verify returned document IDs or answer fields
- file tasks should verify file contents, paths, and read/write effects
- scheduler tasks should verify created jobs and validation outcomes
- CRUD tasks should verify target entity state and untouched distractor state

Transcript checks should remain available, but should usually supplement rather than replace semantic checks.

### 3. Competent Reference Baselines

v1 baselines are intentionally weak reference floors. For publication, v2 needs at least one baseline that can parse task instructions and execute the obvious clean path.

Recommended baseline stack:

- heuristic floor: preserves v1 comparability
- instruction-parser baseline: extracts IDs, paths, cron expressions, values, and reasons
- schema-repair baseline: repairs argument names and types
- policy-aware baseline: avoids or reroutes unsafe/denied actions
- optional local/model baseline: adapter for OpenAI-compatible, Ollama, or llama.cpp style tool callers

This makes failures attributable to misuse and recovery rather than inability to read the task.

### 4. Real Budget Sweeps

v1 reports budgeted success at caps 4, 8, 16, and 32, but generated tasks use a fixed max tool-call budget of 6.

v2 should support real budget profiles:

- tight: 2-3 calls
- standard: 4-6 calls
- relaxed: 8-12 calls
- long-horizon: 16+ calls

The evaluation harness should be able to re-run the same task under different caps so the budgeted success curve measures efficiency rather than a static artifact.

### 5. Cleaner Fault Semantics

v1 fault injection is deterministic and useful, but some semantics should be made sharper.

v2 should separate:

- visible schema drift: changed tool schema is exposed to the agent
- hidden backend drift: old-looking schema but changed backend validation
- transient execution failures: retry can succeed
- persistent failures: retry alone cannot succeed
- recoverable authz: token/role can be supplied or fallback used
- unrecoverable authz: agent must stop safely

This gives v2 a stronger unique selling point: not just "faults happen," but "fault recoverability is explicitly typed."

### 6. Crossed Evaluation Slices

v2 should report results by independent slices:

- domain
- fault type
- difficulty
- budget profile
- recoverable vs unrecoverable fault
- clean vs faulted
- single-fault vs multi-fault

This prevents aggregate scores from hiding narrow failures.

### 7. Proper Ablation Suite

v2 should include first-class ablations:

- no faults
- one fault family at a time
- all faults mixed
- adversarial messages on/off
- schema drift visible vs hidden
- budget relaxed vs strict
- repair disabled vs enabled
- policy layer disabled vs enabled

The ablation config should generate distinct artifacts, not only a different output directory.

### 8. Expanded Quality Report

v1 reports useful uniqueness and balance statistics. v2 should add publication-grade dataset diagnostics:

- domain x fault matrix
- domain x difficulty matrix
- fault x difficulty matrix
- operation x fault matrix
- duplicate instruction clusters
- near-duplicate instruction clusters
- success-criteria distribution
- budget distribution
- recoverability distribution
- checksum verification

These diagnostics should be included in release artifacts and referenced in the paper.

### 9. Threat Model and Dataset Card

v2 should include a complete threat model and dataset card.

Must cover:

- intended use
- out-of-scope use
- synthetic-environment limits
- what the benchmark does not measure
- leakage risks
- overfitting risks
- scoring caveats
- reproducibility guarantees
- versioning policy

This improves credibility for dataset/resource tracks.

### 10. Publication Positioning

v2 should keep the unique selling point narrow and strong:

ToolMisuseBench is not a general agent benchmark. It is a deterministic reliability microscope for tool misuse, typed recoverability, and budgeted recovery behavior.

The v2 paper should emphasize:

- deterministic replay rather than live-service realism
- recoverability typing rather than generic failure injection
- budgeted recovery rather than unlimited retry success
- dataset and harness transparency rather than hidden judge evaluation
- compatibility with stronger external agents rather than benchmark-specific agents

## Suggested v2 Release Shape

Recommended artifact set:

- `data/toolmisusebench_v0_2/`
- `v0_2_freeze.json`
- `manifest.json`
- `quality_report.json`
- `domain_fault_matrix.csv`
- `fault_difficulty_matrix.csv`
- `results.json`
- `tables.md`
- trace JSONL files for all baselines
- dataset card
- threat model

## Non-Goals

- Do not broaden into web browsing, GUI agents, or live APIs.
- Do not rely on paid external services for core reproducibility.
- Do not hide success behind an opaque LLM judge.
- Do not break v1 compatibility unless the change directly fixes a v1 weakness.

