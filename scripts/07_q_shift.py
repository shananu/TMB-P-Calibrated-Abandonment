"""E8: frozen model + frozen conditional-CRC threshold under q shift (train/calibrate at q=0.5, test at q=0.2 / 0.8).
Reads existing data/tmbp_pi0.5_q*/episodes.pkl (read-only). Writes only to results/q_shift/."""
import hashlib, json, pickle, platform, sys, time
import _bootstrap  # noqa: F401  (paths, cwd)
from _bootstrap import RESULTS
from pathlib import Path
import numpy as np, sklearn
from tmbp.shift import CachedScorer, check_aligned, fit_models, make_splits, run_split

CONFIG = {"source_q": "0.5", "target_qs": ["0.2", "0.8"], "pi": 0.5, "alpha": 0.1, "risk": "conditional (FAR_cond)",
          "n_splits": 50, "split_seed": 20261109, "calib_fraction": 0.5, "model": "HistGradientBoosting(max_iter=200, lr=0.05, max_leaf_nodes=15, random_state=0)",
          "table": "Beta posterior per (tool, code, min(consec_fail,4), budget_total)", "tmbp_gen_seed": 7, "tmb_commit": "b6ee341"}
OUT = RESULTS / "q_shift"; OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*.json"):
    sys.exit(f"refusing to overwrite existing {f}; move it away first")

def load(q):
    p = Path(f"data/tmbp_pi0.5_q{q}/episodes.pkl")
    b = p.read_bytes()
    return pickle.loads(b), hashlib.md5(b).hexdigest()
t0 = time.time()
src, src_md5 = load(CONFIG["source_q"])
tgts = {q: load(q) for q in CONFIG["target_qs"]}
for q, (d, _) in tgts.items():
    check_aligned(src["pool"], d["pool"]); check_aligned(src["train"], d["train"])
models = {k: CachedScorer(m) for k, m in fit_models(src["train"], {q: d["train"] for q, (d, _) in tgts.items()}).items()}
splits = make_splits(len(src["pool"]), CONFIG["n_splits"], CONFIG["split_seed"])
raw = [run_split(models, src["pool"], {q: d["pool"] for q, (d, _) in tgts.items()}, ci, ti, CONFIG["alpha"]) for ci, ti in splits]

POL = ["frozen_learned", "frozen_table", "retry_3", "never_stop", "recalibrated_learned", "retrained_recalibrated_learned"]
MET = ["FAR_cond", "success", "mean_calls", "waste_reduction", "FAR", "stop_rate"]
summary = {"lambda_frozen_learned": float(np.mean([r["lambda_frozen_learned"] for r in raw])),
           "lambda_frozen_table": float(np.mean([r["lambda_frozen_table"] for r in raw]))}
for q in CONFIG["target_qs"]:
    s = {"counts_mean": {k: float(np.mean([r[q]["counts"][k] for r in raw])) for k in raw[0][q]["counts"]}}
    for p in POL:
        s[p] = {m: {"mean": float(np.mean(v := [r[q][p][m] for r in raw])), "lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975))} for m in MET}
        s[p]["frac_splits_FARcond_gt_alpha"] = float(np.mean([r[q][p]["FAR_cond"] > CONFIG["alpha"] for r in raw]))
    summary[q] = s
meta = {"config": CONFIG, "inputs_md5": {CONFIG["source_q"]: src_md5, **{q: m for q, (_, m) in tgts.items()}},
        "python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__, "runtime_s": round(time.time() - t0, 1)}
(OUT / "config.json").write_text(json.dumps(meta, indent=1))
(OUT / "splits.json").write_text(json.dumps([{"calib_idx": c.tolist(), "test_idx": t.tolist()} for c, t in splits]))
(OUT / "per_split.json").write_text(json.dumps(raw, indent=1))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))

lines = [f"# E8 q-shift (train+calibrate q={CONFIG['source_q']}, alpha={CONFIG['alpha']}, {CONFIG['n_splits']} splits, seed {CONFIG['split_seed']})\n",
         f"mean frozen lambda: learned {summary['lambda_frozen_learned']:.4f}, table {summary['lambda_frozen_table']:.4f}\n"]
for q in CONFIG["target_qs"]:
    c = summary[q]["counts_mean"]
    lines += [f"\n## target q={q}  (test episodes {c['n_test']:.0f}; counterfactually successful Y=1: {c['n_Y1']:.1f}; Y=1 with >=1 failure: {c['n_Y1_with_failure']:.1f})\n",
              "| policy | FAR_cond mean [2.5%, 97.5%] | splits FAR_cond>alpha | success | mean calls | waste reduction |\n|---|---|---|---|---|---|\n"]
    for p in POL:
        v = summary[q][p]
        lines.append(f"| {p} | {v['FAR_cond']['mean']:.3f} [{v['FAR_cond']['lo']:.3f}, {v['FAR_cond']['hi']:.3f}] | {v['frac_splits_FARcond_gt_alpha']:.2f} | "
                     f"{v['success']['mean']:.3f} | {v['mean_calls']['mean']:.2f} | {v['waste_reduction']['mean']:.3f} |\n")
(OUT / "TABLE.md").write_text("".join(lines), encoding="utf-8")
print("".join(lines)); print(f"runtime {time.time()-t0:.0f}s")
