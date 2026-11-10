"""All TMB-P experiments. Usage: python scripts/03_experiments.py"""
import json, pickle, subprocess, sys, time
import _bootstrap  # noqa: F401  (paths, cwd)
from _bootstrap import RESULTS, guard
from pathlib import Path
from tmbp.experiments import run_policies
from tmbp.generate import FAMILIES

R = RESULTS
guard(*(R / f"{n}.json" for n in ("E1_main_E7_leakage", "E2_budget", "E3_pi", "E4_q", "E5_heldout_family", "E6_heldout_domain")))
def load(pi, q):
    d = Path(f"data/tmbp_pi{pi}_q{q}")
    if not (d / "episodes.pkl").exists():
        subprocess.run([sys.executable, "scripts/02_build_tmbp.py", str(pi), str(q)], check=True)
    with open(d / "episodes.pkl", "rb") as fh:
        eps = pickle.load(fh)
    dom = {}
    for s in ("train", "pool"):
        for line in open(d / f"{s}.jsonl", encoding="utf-8"):
            t = json.loads(line); dom[t["task_id"]] = t["domain"]
    for s in eps.values():
        for e in s: e.gold["domain"] = dom[e.task_id]
    return eps

def dump(name, obj):
    (R / f"{name}.json").write_text(json.dumps(obj, indent=1)); print("wrote", name, flush=True)

t0 = time.time()
eps = load(0.5, 0.5)
# E1 main + E7 leakage ablation (feature sets)
dump("E1_main_E7_leakage", run_policies(eps["train"], eps["pool"], pi=0.5, q=0.5,
     feature_sets=("observable", "leak_fault_type", "leak_recoverability_tag")))
# E2 budget sensitivity (train on all budgets; calibrate + test within each budget profile)
dump("E2_budget", {p: run_policies(eps["train"], [e for e in eps["pool"] if e.gold["profile"] == p], pi=0.5, q=0.5)
                   for p in ("tight", "standard", "relaxed")})
# E3 persistent proportion, E4 transient failure probability
dump("E3_pi", {pi: run_policies(**dict(zip(("train", "pool"), (lambda e: (e["train"], e["pool"]))(load(pi, 0.5)))), pi=pi, q=0.5)
               for pi in (0.1, 0.3, 0.5, 0.7, 0.9)})
dump("E4_q", {q: run_policies(**dict(zip(("train", "pool"), (lambda e: (e["train"], e["pool"]))(load(0.5, q)))), pi=0.5, q=q)
              for q in (0.2, 0.5, 0.8)})
# E5 held-out fault family, E6 held-out tool domain (calibration on in-distribution data only)
def held_out(key, values):
    out = {}
    for v in values:
        out[v] = run_policies([e for e in eps["train"] if e.gold[key] != v],
                              [e for e in eps["pool"] if e.gold[key] == v], pi=0.5, q=0.5,
                              calib_pool=[e for e in eps["pool"] if e.gold[key] != v], n_splits=20)
    return out
dump("E5_heldout_family", held_out("family", FAMILIES))
dump("E6_heldout_domain", held_out("domain", ("crud", "retrieval", "files", "scheduling")))
print(f"done in {time.time()-t0:.0f}s")
