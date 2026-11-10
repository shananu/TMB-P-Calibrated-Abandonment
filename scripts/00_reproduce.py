"""Step 0: regenerate the seed-42 v2 'large' release and reproduce reference-agent numbers."""
import json, sys
import _bootstrap  # noqa: F401  (paths, cwd)
from _bootstrap import RESULTS, guard
from pathlib import Path
from toolmisusebench.version2.generator import generate_dataset_v2
from toolmisusebench.dataset import load_tasks
from toolmisusebench.baselines import build_agent
from toolmisusebench.harness.runner import run_episode

OUT = RESULTS / "00_reproduce.json"
guard(OUT)
root = Path("data/tmb_v02_large")
if not (root / "manifest.json").exists():
    print(generate_dataset_v2("v0.2", root, seed=42, size_profile="large"))
tasks = load_tasks(root, "test_public")
print("n test_public:", len(tasks))
out = {}
for name in ["instruction_parser", "instruction_parser_no_recovery"]:
    rs = [run_episode(t, build_agent(name)) for t in tasks]
    faulted = [r for r in rs if r.primary_fault != "clean"]
    seen = [r for r in faulted if r.fault_seen]
    out[name] = {
        "task_success": sum(r.success for r in rs) / len(rs),
        "cond_recovery": sum(r.success for r in seen) / max(1, len(seen)),
        "exposure": len(seen) / max(1, len(faulted)),
        "mean_calls": sum(r.tool_calls for r in rs) / len(rs),
    }
    print(name, json.dumps(out[name]))
OUT.write_text(json.dumps(out, indent=2))
