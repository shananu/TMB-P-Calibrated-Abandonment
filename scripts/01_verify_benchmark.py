"""Step 1: empirically verify the audit's static claims about ToolMisuseBench (commit b6ee341)."""
import json
import _bootstrap  # noqa: F401  (paths, cwd)
from _bootstrap import RESULTS, guard
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from toolmisusebench.dataset import load_tasks
from toolmisusebench.baselines import build_agent
from toolmisusebench.harness.runner import run_episode, make_env_for_task
from toolmisusebench.types import Task

OUT = RESULTS / "01_verify.json"
guard(OUT)
root = Path("data/tmb_v02_large")
report = {}
all_tasks = [t for s in ("train", "dev", "test_public") for t in load_tasks(root, s)]

# (1) one-shot fault generation
triggers = Counter()
tag_by_family = defaultdict(Counter)
for t in all_tasks:
    for i, f in enumerate(t.fault_plan):
        triggers[(f.fault_type, json.dumps(f.trigger.get("on_nth_call")), "prob" in f.trigger)] += 1
    fam = t.fault_plan[0].fault_type if t.fault_plan else "clean"
    tag = t.fault_plan[0].payload.get("recoverability") if t.fault_plan else None
    tag_by_family[fam][str(tag)] += 1
report["n_tasks"] = len(all_tasks)
report["trigger_counts(fault_type,on_nth_call,has_prob)"] = {str(k): v for k, v in triggers.items()}
report["recoverability_tag_by_primary_family"] = {k: dict(v) for k, v in tag_by_family.items()}

# (2) label leakage: what does the agent observe after a faulted first call?
leak = Counter()
leak_examples = {}
for t in load_tasks(root, "test_public"):
    if not t.fault_plan:
        continue
    env = make_env_for_task(t)
    env.reset(t)
    agent = build_agent("instruction_parser")
    agent.reset()
    obs = env.observation()
    a = agent.act(obs)
    env.step(a)
    obs = env.observation()
    err = obs.last_error
    faults = (err.details.get("faults", []) if err else [])
    visible = sorted({f.get("payload", {}).get("recoverability") for f in faults if f.get("payload", {}).get("recoverability")})
    visible_types = sorted({f.get("fault_type") for f in faults})
    fam = t.fault_plan[0].fault_type
    leak[(fam, tuple(visible), tuple(visible_types))] += 1
    leak_examples.setdefault(fam, err.model_dump() if err else None)
report["observed_after_first_call(family,visible_recoverability,visible_fault_type)"] = {str(k): v for k, v in leak.items()}
report["first_error_examples"] = leak_examples

# (3) persistent / probabilistic triggers fire as expected
def mk(task, plan):
    d = task.model_dump(); d["fault_plan"] = plan
    d["budget"] = {"max_steps": 12, "max_tool_calls": 12, "max_retries": 11, "timeout_ms": 1500}
    return Task.model_validate(d)
base = next(t for t in load_tasks(root, "test_public") if t.fault_plan and t.fault_plan[0].fault_type == "timeout")
tool = base.fault_plan[0].trigger["on_tool"]
persist = mk(base, [{"fault_type": "timeout", "trigger": {"on_tool": tool}, "severity": "soft", "payload": {"timeout_ms": 50}}])
r = run_episode(persist, build_agent("instruction_parser"))
report["persistent_timeout"] = {"success": r.success, "tool_calls": r.tool_calls,
                                "error_codes": [e["error"]["code"] if e["error"] else None for e in r.transcript]}
probs = []
for seed in range(5):
    pt = mk(base, [{"fault_type": "timeout", "trigger": {"on_tool": tool, "prob": {"p": 0.6, "seed": 1000 + seed}}, "severity": "soft", "payload": {"timeout_ms": 50}}])
    rr = run_episode(pt, build_agent("instruction_parser"))
    probs.append([e["error"]["code"] if e["error"] else "ok" for e in rr.transcript])
report["prob_timeout_p0.6_5seeds"] = probs

# (4) deterministic replay
mism = 0
for t in load_tasks(root, "test_public")[:600]:
    a = run_episode(t, build_agent("instruction_parser")); b = run_episode(t, build_agent("instruction_parser"))
    mism += int(a.transcript != b.transcript or a.success != b.success)
report["replay_mismatches_first600"] = mism

# (5) why does Parser+Repair fail on the released test split?
fails = Counter(); tot = Counter()
for t in load_tasks(root, "test_public"):
    r = run_episode(t, build_agent("instruction_parser"))
    prof = t.gold_summary.split(";")[0].split("=")[1]
    key = (r.primary_fault, prof)
    tot[key] += 1
    if not r.success:
        last = r.transcript[-1]["error"]["code"] if r.transcript and r.transcript[-1]["error"] else "agent_stopped_no_error"
        fails[(r.primary_fault, prof, last)] += 1
report["parser_repair_failures(family,profile,last_error)"] = {str(k): v for k, v in sorted(fails.items())}
report["totals(family,profile)"] = {str(k): v for k, v in sorted(tot.items())}
OUT.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=1, default=str)[:9000])
