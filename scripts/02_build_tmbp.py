"""Build TMB-P (default pi=0.5, q=0.5), roll out never-stop, and verify exact-replay labels."""
import json, pickle, sys, time
import _bootstrap  # noqa: F401  (paths, cwd)
from _bootstrap import guard
from collections import defaultdict
from pathlib import Path
from tmbp.generate import build_tmbp, save
from tmbp.core import rollout, evaluate

pi, q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5, float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
NTR, NPOOL = 2160, 2160
t0 = time.time()
tasks = build_tmbp(pi, q, NTR, NPOOL)
out = Path(f"data/tmbp_pi{pi}_q{q}")
guard(out / "episodes.pkl")
save(tasks, out, {"pi": pi, "q": q, "n_train": NTR, "n_pool": NPOOL, "gen_seed": 7, "tmb_commit": "b6ee341"})
eps = {s: [rollout(t)[0] for t in ts] for s, ts in tasks.items()}
pickle.dump(eps, open(out / "episodes.pkl", "wb"))
print(f"built+rolled {sum(map(len, tasks.values()))} episodes in {time.time()-t0:.1f}s")

# outcome table on the pool
tab = defaultdict(lambda: [0, 0, 0])
for e in eps["pool"]:
    k = (e.gold["family"], e.gold["mode"], e.gold["profile"])
    tab[k][0] += 1; tab[k][1] += e.y; tab[k][2] += e.calls_never_stop
summary = {str(k): {"n": v[0], "Y_rate": round(v[1]/v[0], 3), "calls": round(v[2]/v[0], 2)} for k, v in sorted(tab.items())}

# sanity 1: stopping is never "free success"
bad = sum(dp.success_if_stop for s in eps.values() for e in s for dp in e.dps)
# sanity 2: analytic prefix evaluation == actual wrapper rollouts for retry-k rules
mism = 0; checked = 0
for k in (0, 1, 2):
    rule = lambda f, k=k: f["consec_fail"] > k
    for t, e in list(zip(tasks["pool"], eps["pool"]))[:720]:
        ep_real, res = rollout(t, stop_fn=rule)
        an = evaluate([e], [[0.0 if rule(dp.feats) else 1.0 for dp in e.dps]], 0.5)
        mism += int(an["success"] != res.success or abs(an["mean_calls"] - res.tool_calls) > 1e-9)
        checked += 1
rep = {"pi": pi, "q": q, "decision_points_with_success_already": bad, "replay_checks": checked, "replay_mismatches": mism,
       "pool_Y_rate": sum(e.y for e in eps["pool"]) / len(eps["pool"]), "outcomes": summary}
(out / "build_report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k != "outcomes"}))
for k, v in summary.items(): print(k, v)
