"""Where do the savings come from? Per-family breakdown on one fixed calib/test split (pi=0.5,q=0.5)."""
import json, pickle
import _bootstrap  # noqa: F401  (paths, cwd)
from _bootstrap import RESULTS, guard
import numpy as np
from tmbp.experiments import Learned, rule_scores
from tmbp.core import crc_threshold, evaluate
guard(RESULTS / "05_breakdown.json")
with open("data/tmbp_pi0.5_q0.5/episodes.pkl", "rb") as fh:
    eps = pickle.load(fh)
pool = eps["pool"]; rng = np.random.default_rng(123); perm = rng.permutation(len(pool)); h = len(pool)//2
cal = [pool[i] for i in perm[:h]]; test = [pool[i] for i in perm[h:]]
m = Learned().fit(eps["train"])
lam = crc_threshold(cal, m.scores(cal), 0.1, conditional=True)
out = {"lambda": lam}
for fam in ["timeout", "rate_limit", "schema_drift", "authz", "adversarial_error"]:
    for mode in ["persistent", "transient"]:
        sub = [e for e in test if e.gold["family"] == fam and e.gold["mode"] == mode]
        a = evaluate(sub, m.scores(sub), lam); r3 = evaluate(sub, rule_scores(sub, lambda f: f["consec_fail"] > 3), .5)
        out[f"{fam}/{mode}"] = {"n": len(sub), "Y": round(np.mean([e.y for e in sub]), 3),
            "learned_calls": round(a["mean_calls"], 2), "learned_FARcond": round(a["FAR_cond"], 3),
            "retry3_calls": round(r3["mean_calls"], 2), "retry3_FARcond": round(r3["FAR_cond"], 3),
            "never_stop_calls": round(np.mean([e.calls_never_stop for e in sub]), 2)}
with open(RESULTS / "05_breakdown.json", "w") as fh:
    json.dump(out, fh, indent=1)
for k, v in out.items(): print(k, v)
