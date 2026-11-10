"""TMB-P: ToolMisuseBench extended with persistent and probabilistic-transient faults.

Each TMB-P task reuses a frozen ToolMisuseBench v2 scenario (instruction, initial state,
tool schemas, output-grounded success criteria, fault tool) and replaces only:
  * the fault plan  (persistent vs. probabilistic-transient, mixed with proportion pi)
  * the budget      (tight/standard/relaxed -> 4/8/12 calls; retry cap never binds first)

Fault semantics (all use the unmodified ToolMisuseBench fault engine):
  persistent : the fault fires on EVERY call to the fault tool.
  transient  : the fault fires on the FIRST call (so every faulted episode is exposed),
               then independently with probability q on each later call (seeded).
Under 'transient' the error signal is identical to 'persistent' as long as the fault keeps
firing (in particular at the first failure, which every faulted episode reaches). After a call on
which the transient fault does not fire the agent may succeed, or - for schema drift - receive the
environment's genuine error for its drift-repaired arguments (e.g. 'unknown_argument'); both are
legitimate observable signal, not leaked metadata.

Ground truth (mode, family) is stored in `gold_summary` and the fault payload; it is
NEVER used as a predictor feature (see observable_features in tmbp/core.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any

from toolmisusebench.types import Budget, Task
from toolmisusebench.version2.generator import _build_task_v2, _rename_args_for_tool

BUDGETS = {
    "tight": Budget(max_steps=4, max_tool_calls=4, max_retries=4, timeout_ms=1000),
    "standard": Budget(max_steps=8, max_tool_calls=8, max_retries=8, timeout_ms=1000),
    "relaxed": Budget(max_steps=12, max_tool_calls=12, max_retries=12, timeout_ms=1000),
}
FAMILIES = ("timeout", "rate_limit", "schema_drift", "authz", "adversarial_error")

# TMB-P split -> underlying v2 scenario split (scenario offsets differ per v2 split)
SOURCE_SPLIT = {"train": "train", "pool": "test_public"}


def _base_fault(family: str, tool: str, trigger: dict[str, Any], tag: str | None) -> dict[str, Any]:
    payload: dict[str, Any]
    if family == "timeout":
        payload, sev = {"timeout_ms": 50}, "soft"
    elif family == "rate_limit":
        payload, sev = {"after_n_calls": 0, "retry_after_ms": 100}, "soft"
    elif family == "authz":
        payload, sev = {"deny": True, "required_role": "admin"}, "hard"
    elif family == "schema_drift":
        payload, sev = {"rename_args": _rename_args_for_tool(tool), "drift_visibility": "hidden_backend"}, "hard"
    else:
        raise ValueError(family)
    if tag is not None:
        payload["recoverability"] = tag
    return {"fault_type": family, "trigger": {"on_tool": tool, **trigger}, "severity": sev, "payload": payload}


def fault_plan(family: str, tool: str, mode: str, q: float, fault_seed: int) -> list[dict[str, Any]]:
    """Build a fault plan. The first spec carries the ground-truth recoverability tag,
    mirroring ToolMisuseBench's typed recoverability metadata (and therefore its leak)."""
    tag = "persistent" if mode == "persistent" else "transient"
    inner = "timeout" if family == "adversarial_error" else family
    if mode == "persistent":
        plan = [_base_fault(inner, tool, {}, tag)]
    else:
        plan = [
            _base_fault(inner, tool, {"on_nth_call": 1}, tag),
            _base_fault(inner, tool, {"prob": {"p": q, "seed": fault_seed}}, None),
        ]
    if family == "adversarial_error":
        # rewrite every error on this tool to an opaque code/message (as in the v2 release)
        plan.insert(0, {
            "fault_type": "adversarial_error",
            "trigger": {"on_tool": tool},
            "severity": "soft",
            "payload": {"message": "The tool failed for an unclear reason. Try a safer recovery.",
                        "code": "ambiguous_error", "recoverability": tag},
        })
    return plan


def build_task(split: str, index: int, pi: float, q: float, seed: int) -> Task:
    base = _build_task_v2(split=SOURCE_SPLIT[split], index=index, base_seed=42)
    gs = dict(kv.strip().split("=") for kv in base.gold_summary.split(";"))
    profile, family = gs["v2_profile"], gs["fault_family"]
    rng = Random(seed * 1_000_003 + index * 7919 + (0 if split == "train" else 500_009))
    d = base.model_dump()
    d["budget"] = BUDGETS[profile].model_dump()
    if family == "clean":
        mode = "none"
        d["fault_plan"] = []
    else:
        tool = base.fault_plan[0].trigger["on_tool"]
        mode = "persistent" if rng.random() < pi else "transient"
        d["fault_plan"] = fault_plan(family, tool, mode, q, fault_seed=rng.randrange(1 << 30))
    d["split"] = "train" if split == "train" else "test_public"
    d["task_id"] = f"tmbp_{split}_{index:05d}"
    d["gold_summary"] = f"profile={profile}; family={family}; mode={mode}; pi={pi}; q={q}"
    d["seed"] = seed * 100_000 + index + (0 if split == "train" else 50_000)
    return Task.model_validate(d)


def parse_gold(task: Task) -> dict[str, str]:
    return dict(kv.strip().split("=") for kv in task.gold_summary.split(";"))


def build_tmbp(pi: float, q: float, n_train: int, n_pool: int, seed: int = 7) -> dict[str, list[Task]]:
    return {
        "train": [build_task("train", i, pi, q, seed) for i in range(n_train)],
        "pool": [build_task("pool", i, pi, q, seed) for i in range(n_pool)],
    }


def save(tasks: dict[str, list[Task]], out: Path, meta: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in tasks.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for t in rows:
                fh.write(json.dumps(t.model_dump(), sort_keys=True) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
