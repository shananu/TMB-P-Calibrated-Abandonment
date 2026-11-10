# ToolMisuseBench v2 Threat Model

## What v2 Measures

ToolMisuseBench v2 measures deterministic behavior of tool-using agents under controlled tool-interface and execution failures. It focuses on whether an agent can:

- construct valid tool calls
- recover after structured tool failures
- avoid repeated invalid calls
- respect step, retry, and tool-call budgets
- handle ambiguous or adversarial error messages
- continue to solve tasks when faults are recoverable

## What v2 Does Not Measure

ToolMisuseBench v2 does not directly measure:

- real service reliability
- real authentication and authorization systems
- network jitter or distributed-system failure distributions
- monetary cost of tool calls
- human preference alignment
- broad open-ended planning
- malicious exploitation capability

## Synthetic Environment Risk

The benchmark uses compact deterministic simulators. This improves reproducibility and attribution, but limits external validity. Results should be interpreted as controlled reliability evidence, not as proof of production readiness.

## Overfitting Risk

Because tasks are generated from visible templates, agents can overfit to instruction patterns. v2 mitigates this by:

- crossing domain, fault, difficulty, and budget slices
- reporting instruction uniqueness
- supporting regenerated splits with fixed seeds
- exposing per-slice metrics rather than only aggregate scores

Future releases should add near-duplicate detection and additional template families.

## Leakage Risk

The benchmark includes public train, development, and public test splits. If public test examples are used for agent development, reported performance may overstate generalization. Publication results should identify the exact split and seed, and ideally include regenerated held-out splits.

## Scoring Caveats

Output-grounded checks improve task-completion validity, but they are still simulator-specific. A score improvement can reflect better adaptation to benchmark semantics rather than general tool-use reliability. Per-domain and per-fault reporting is required for meaningful interpretation.

## Policy Caveats

The current policy layer is intentionally lightweight. Policy-violation counts should be interpreted as benchmark-defined violation signals, not as comprehensive safety guarantees.

## Recommended Reporting

Papers using v2 should report:

- aggregate metrics
- domain-specific metrics
- fault-specific metrics
- fault by difficulty metrics
- budgeted success curves
- recovery success and time to recovery
- dataset version, seed, split sizes, and checksum manifest

