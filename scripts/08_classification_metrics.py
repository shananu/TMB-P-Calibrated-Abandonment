"""Classification metrics for the recovery-success predictor (main setting: TMB-P pi=0.5, q=0.5).

WHAT IS CLASSIFIED
  Unit      : one decision point = an observation immediately after a failed tool call.
  Label     : Y = 1 if letting the base agent continue (never stop) ends in task success,
              Y = 0 otherwise. Y is the episode's exact counterfactual "continue" outcome, shared
              by every decision point of that episode.
  Prediction: p_hat = P(Y=1 | observable features), observable feature set only (no leak_* fields).

DATA
  Train : data/tmbp_pi0.5_q0.5 train episodes (v2 'train' scenarios) - the ONLY data used to fit.
  Test  : all 2,160 pool episodes (v2 'test_public' scenarios), never used for fitting. This is the
          exact set behind the saved AUROC 0.870 (results/E1_main_E7_leakage.json,
          'predictor_diagnostics_on_test_pool'). Note: pool halves were used to pick the CRC STOPPING
          threshold in E1, never for classification or model fitting.

NO SAVED PER-EXAMPLE PREDICTIONS EXIST. Models are refit with the identical deterministic code on
train only, and the script asserts that the refit reproduces the saved AUROC/Brier exactly.

THRESHOLDS (never chosen on test labels)
  T_0.5   : pre-specified 0.5 cut-off on p_hat (sklearn default; the 'learned_*@0.5' policies).
  T_train : chosen on TRAINING data only - 5-fold GroupKFold (groups = episodes) out-of-fold
            p_hat on the train split, threshold maximising balanced accuracy (Youden's J).
  lambda  : the CRC stopping threshold (conditional, alpha=0.1) is a different object - a risk-
            controlled cut-off for abandoning episodes. Reported separately, on E1's own 50
            calibration/test halvings (lambda from the calibration half, metrics on the test half).
"""
import json, pickle, sys
import _bootstrap  # noqa: F401  (paths, cwd, utf-8 stdout)
from _bootstrap import RESULTS
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold

from tmbp.core import crc_threshold
from tmbp.experiments import FEATURE_SETS, Learned, dp_table

OUT_JSON, OUT_MD, FIG = RESULTS / "classification_metrics.json", RESULTS / "CLASSIFICATION_METRICS.md", RESULTS / "figures"
for p in (OUT_JSON, OUT_MD, FIG / "fig4_confusion_matrices.png", FIG / "fig5_roc_pr.png"):
    if p.exists():
        sys.exit(f"refusing to overwrite existing {p}")
SAVED = json.loads((RESULTS / "E1_main_E7_leakage.json").read_text())["predictor_diagnostics_on_test_pool"]
N_BOOT, BOOT_SEED, CV_SEED_NOTE = 1000, 20261110, "GroupKFold is deterministic (no shuffling)"

with open("data/tmbp_pi0.5_q0.5/episodes.pkl", "rb") as fh:
    eps = pickle.load(fh)
train, pool = eps["train"], eps["pool"]
assert not {e.task_id for e in train} & {e.task_id for e in pool}, "train/test overlap"
assert not any(c.startswith("leak") for c in FEATURE_SETS["observable"][0] + FEATURE_SETS["observable"][1])

MODELS = {"gradient_boosting": "hgb", "logistic_regression": "lr"}
SAVED_KEY = {"gradient_boosting": "observable", "logistic_regression": "learned_lr"}


def episode_index(e_list):
    return np.array([i for i, e in enumerate(e_list) for _ in e.dps])


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {"threshold": float(thr), "accuracy": float((tp + tn) / len(y)), "precision": float(prec),
            "recall_sensitivity": float(rec), "specificity": float(spec), "npv": float(npv),
            "f1_positive": float(f1_score(y, yhat, pos_label=1, zero_division=0)),
            "f1_negative": float(f1_score(y, yhat, pos_label=0, zero_division=0)),
            "f1_macro": float(f1_score(y, yhat, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y, yhat, average="weighted", zero_division=0)),
            "balanced_accuracy": float((rec + spec) / 2),
            "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "auprc_negative_class": float(average_precision_score(1 - y, 1 - p)),
            "brier": float(brier_score_loss(y, p)),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def cluster_bootstrap(y, p, groups, thr, keys=("accuracy", "precision", "recall_sensitivity", "specificity",
                                                "f1_positive", "f1_macro", "balanced_accuracy", "auroc", "auprc", "brier")):
    """95% CI resampling EPISODES (decision points within an episode share a label)."""
    rng = np.random.default_rng(BOOT_SEED)
    uniq = np.unique(groups)
    by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = {k: [] for k in keys}
    for _ in range(N_BOOT):
        idx = np.concatenate([by_g[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        m = metrics(y[idx], p[idx], thr)
        for k in keys:
            vals[k].append(m[k])
    return {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in vals.items()}


def train_selected_threshold(kind):
    """Youden-J threshold from out-of-fold predictions on TRAIN only (groups = episodes)."""
    rows, y = dp_table(train)
    g = episode_index(train)
    oof = np.zeros(len(y))
    for tr_i, va_i in GroupKFold(n_splits=5).split(rows, y, g):
        sub = sorted(set(g[tr_i]))
        m = Learned("observable", model=kind).fit([train[i] for i in sub])
        oof[va_i] = m.model.predict_proba(m.enc.transform([rows[i] for i in va_i]))[:, 1]
    fpr, tpr, thr = roc_curve(y, oof)
    j = np.argmax(tpr - fpr)
    return float(thr[j]), float(roc_auc_score(y, oof))


results = {"definition": {
    "unit": "decision point (observation after a failed tool call)",
    "positive_class": "Y=1: continuing with the base agent ends in task success (episode-level counterfactual, shared by all its decision points)",
    "negative_class": "Y=0: continuing does not succeed (stopping would only save calls)",
    "train": "TMB-P pi=0.5 q=0.5 train split (v2 train scenarios)",
    "test": "all 2,160 TMB-P pi=0.5 q=0.5 pool episodes (v2 test_public scenarios); never used for fitting or classification thresholds",
    "features": "observable set only: " + ", ".join(FEATURE_SETS["observable"][0] + FEATURE_SETS["observable"][1]),
    "ci": f"95% cluster bootstrap over episodes, {N_BOOT} resamples, seed {BOOT_SEED}",
    "threshold_selection": "T_0.5 pre-specified; T_train = Youden J on 5-fold GroupKFold out-of-fold train predictions (" + CV_SEED_NOTE + ")"},
    "models": {}}

rows_te, y_te = dp_table(pool)
g_te = episode_index(pool)
first_mask = np.r_[True, g_te[1:] != g_te[:-1]]  # first decision point of each episode
rows_tr, y_tr = dp_table(train)
results["support"] = {
    "train_decision_points": {"n": len(y_tr), "Y1": int(y_tr.sum()), "Y0": int(len(y_tr) - y_tr.sum()), "episodes_with_dp": int(sum(bool(e.dps) for e in train))},
    "test_decision_points": {"n": len(y_te), "Y1": int(y_te.sum()), "Y0": int(len(y_te) - y_te.sum()), "episodes_with_dp": int(sum(bool(e.dps) for e in pool)),
                             "episodes_total": len(pool), "episodes_without_dp (never fail, never classified)": int(sum(not e.dps for e in pool))},
    "test_first_decision_points": {"n": int(first_mask.sum()), "Y1": int(y_te[first_mask].sum()), "Y0": int(first_mask.sum() - y_te[first_mask].sum())}}

preds = {}
for name, kind in MODELS.items():
    m = Learned("observable", model=kind).fit(train)
    p = m.model.predict_proba(m.enc.transform(rows_te))[:, 1]
    preds[name] = p
    # alignment / reproduction check against the saved E1 diagnostics
    saved = SAVED[SAVED_KEY[name]]
    assert abs(roc_auc_score(y_te, p) - saved["auroc"]) < 1e-12 and abs(brier_score_loss(y_te, p) - saved["brier"]) < 1e-12, \
        f"{name}: refit does not reproduce saved AUROC/Brier"
    t_train, oof_auc = train_selected_threshold(kind)
    r = {"reproduces_saved_E1": {"auroc": saved["auroc"], "brier": saved["brier"]}, "train_oof_auroc": oof_auc}
    for tname, thr in (("T_0.5", 0.5), ("T_train", t_train)):
        r[tname] = {"all_decision_points": metrics(y_te, p, thr), "ci95": cluster_bootstrap(y_te, p, g_te, thr),
                    "first_decision_point_only": metrics(y_te[first_mask], p[first_mask], thr)}
    # CRC stopping threshold, E1 protocol: rng=default_rng(0), 50 permutations, calib = first half
    rng = np.random.default_rng(0); per = []
    sc = m.scores(pool)
    for _ in range(50):
        perm = rng.permutation(len(pool)); h = len(pool) // 2
        cal, te = perm[:h], perm[h:]
        lam = crc_threshold([pool[i] for i in cal], [sc[i] for i in cal], 0.1, conditional=True)
        mask = np.isin(g_te, te)
        per.append(metrics(y_te[mask], p[mask], lam))
    keys = [k for k in per[0] if k != "confusion"]
    r["CRC_stopping_lambda_alpha0.1"] = {"note": "lambda chosen on each calibration half; metrics on the matching test half; mean over 50 splits",
                                         **{k: float(np.mean([x[k] for x in per])) for k in keys},
                                         "confusion_mean": {c: float(np.mean([x["confusion"][c] for x in per])) for c in ("TN", "FP", "FN", "TP")}}
    results["models"][name] = r

# ---- paired comparison (same test decision points, same episode resamples) ----
def paired_diff(seed=BOOT_SEED + 1):
    rng = np.random.default_rng(seed)
    uniq = np.unique(g_te); by_g = {g: np.flatnonzero(g_te == g) for g in uniq}
    a, b = preds["gradient_boosting"], preds["logistic_regression"]
    ta, tb = results["models"]["gradient_boosting"]["T_train"]["all_decision_points"]["threshold"], results["models"]["logistic_regression"]["T_train"]["all_decision_points"]["threshold"]
    d = {"auroc": [], "auprc": [], "brier": [], "balanced_accuracy@T_train": [], "first_dp_auroc": []}
    for _ in range(N_BOOT):
        idx = np.concatenate([by_g[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        y = y_te[idx]
        d["auroc"].append(roc_auc_score(y, a[idx]) - roc_auc_score(y, b[idx]))
        d["auprc"].append(average_precision_score(y, a[idx]) - average_precision_score(y, b[idx]))
        d["brier"].append(brier_score_loss(y, a[idx]) - brier_score_loss(y, b[idx]))
        d["balanced_accuracy@T_train"].append(metrics(y, a[idx], ta)["balanced_accuracy"] - metrics(y, b[idx], tb)["balanced_accuracy"])
        fm = first_mask[idx]
        d["first_dp_auroc"].append(roc_auc_score(y[fm], a[idx][fm]) - roc_auc_score(y[fm], b[idx][fm]))
    return {k: {"mean": float(np.mean(v)), "ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))]} for k, v in d.items()}

results["paired_difference_gb_minus_lr"] = paired_diff()
for name in MODELS:  # CI for first-decision-point AUROC (one prediction per episode -> ordinary bootstrap)
    rng = np.random.default_rng(BOOT_SEED + 2); yf, pf = y_te[first_mask], preds[name][first_mask]
    v = [roc_auc_score(yf[i], pf[i]) for i in (rng.integers(0, len(yf), len(yf)) for _ in range(N_BOOT))]
    results["models"][name]["first_decision_point_auroc_ci95"] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
OUT_JSON.write_text(json.dumps(results, indent=1))

# ---------------- markdown ----------------
S = results["support"]
L = ["# Classification metrics: recovery-success predictor (TMB-P, pi=0.5, q=0.5)\n\n",
     "**Unit:** decision point (after a failed tool call). **Positive class (Y=1):** continuing would succeed. ",
     "**Test set:** all 2,160 pool episodes, never used for fitting or for choosing a classification threshold. ",
     "Refit models reproduce the saved E1 AUROC/Brier exactly (asserted).\n\n",
     f"**Support (test decision points):** n={S['test_decision_points']['n']}, Y=1: {S['test_decision_points']['Y1']}, Y=0: {S['test_decision_points']['Y0']} "
     f"(from {S['test_decision_points']['episodes_with_dp']} episodes with ≥1 failure; {S['test_decision_points']['episodes_without_dp (never fail, never classified)']} episodes never fail and are never classified). "
     f"Train: n={S['train_decision_points']['n']}, Y=1: {S['train_decision_points']['Y1']}, Y=0: {S['train_decision_points']['Y0']}.\n\n",
     "95% CIs: cluster bootstrap over episodes (1,000 resamples).\n\n"]
cols = [("accuracy", "Accuracy"), ("balanced_accuracy", "Bal. acc."), ("precision", "Precision"), ("recall_sensitivity", "Recall"),
        ("specificity", "Specificity"), ("f1_positive", "F1 (Y=1)"), ("f1_macro", "Macro F1"), ("f1_weighted", "Weighted F1"),
        ("auroc", "AUROC"), ("auprc", "AUPRC"), ("brier", "Brier")]
for tname, title in (("T_0.5", "Threshold 0.5 (pre-specified)"), ("T_train", "Threshold selected on training data (Youden J, out-of-fold)")):
    L.append(f"\n## {title} — all test decision points\n\n| model | thr | " + " | ".join(c[1] for c in cols) + " | TN | FP | FN | TP |\n|" + "---|" * (len(cols) + 6) + "\n")
    for name in MODELS:
        m, ci = results["models"][name][tname]["all_decision_points"], results["models"][name][tname]["ci95"]
        cells = [f"{m[k]:.3f}" + (f" [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if k in ci else "") for k, _ in cols]
        c = m["confusion"]
        L.append(f"| {name} | {m['threshold']:.3f} | " + " | ".join(cells) + f" | {c['TN']} | {c['FP']} | {c['FN']} | {c['TP']} |\n")
    L.append(f"\n**Sensitivity analysis (one prediction per episode — first decision point only; n={S['test_first_decision_points']['n']}, Y=1: {S['test_first_decision_points']['Y1']}):**\n\n"
             "| model | Accuracy | Bal. acc. | Precision | Recall | Specificity | Macro F1 | AUROC | TN | FP | FN | TP |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for name in MODELS:
        m = results["models"][name][tname]["first_decision_point_only"]; c = m["confusion"]
        L.append(f"| {name} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['precision']:.3f} | {m['recall_sensitivity']:.3f} | {m['specificity']:.3f} | "
                 f"{m['f1_macro']:.3f} | {m['auroc']:.3f} | {c['TN']} | {c['FP']} | {c['FN']} | {c['TP']} |\n")
L.append("\n## Operating point of the stopping policy (CRC lambda, alpha=0.1; NOT a classification threshold)\n\n"
         "lambda chosen on each E1 calibration half, metrics on the matching test half, mean over 50 splits.\n\n"
         "| model | mean lambda | Accuracy | Precision (Y=1) | Recall (Y=1) | Specificity | NPV (precision of 'stop') | Mean TN | FP | FN | TP |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
for name in MODELS:
    m = results["models"][name]["CRC_stopping_lambda_alpha0.1"]; c = m["confusion_mean"]
    L.append(f"| {name} | {m['threshold']:.3f} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall_sensitivity']:.3f} | {m['specificity']:.3f} | {m['npv']:.3f} | "
             f"{c['TN']:.1f} | {c['FP']:.1f} | {c['FN']:.1f} | {c['TP']:.1f} |\n")
L.append("\n## Paired comparison, gradient boosting − logistic regression (same test points; episode cluster bootstrap)\n\n| metric | mean difference | 95% CI |\n|---|---|---|\n")
for k, v in results["paired_difference_gb_minus_lr"].items():
    L.append(f"| {k} | {v['mean']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |\n")
L.append("\nFirst-decision-point AUROC 95% CI: " + "; ".join(f"{n} {results['models'][n]['first_decision_point_auroc_ci95'][0]:.3f}–{results['models'][n]['first_decision_point_auroc_ci95'][1]:.3f}" for n in MODELS) + "\n")
L.append("\nNote: the CRC-row confusion counts treat every decision point as an independent classification. The stopping policy acts only at the FIRST decision point "
         "below lambda, so its episode-level FAR_cond (README, E1) - not this table - is the policy's error rate.\n")
OUT_MD.write_text("".join(L), encoding="utf-8")

# ---------------- figures ----------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"gradient_boosting": "#2a78d6", "logistic_regression": "#eb6834"}
LBL = {"gradient_boosting": "Gradient boosting", "logistic_regression": "Logistic regression"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.8))
for ax, name in zip(axs, MODELS):
    c = results["models"][name]["T_0.5"]["all_decision_points"]["confusion"]
    mat = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    ax.imshow(mat / mat.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = mat[i, j] / mat[i].sum()
            ax.text(j, i, f"{mat[i, j]}\n({frac:.1%} of row)", ha="center", va="center", color="#ffffff" if frac > .6 else INK, fontsize=9)
    ax.set_xticks([0, 1], ["pred Y=0\n(stop)", "pred Y=1\n(continue)"]); ax.set_yticks([0, 1], ["true Y=0", "true Y=1"])
    ax.set_title(f"{LBL[name]} (threshold 0.5)", fontsize=10, loc="left", color=INK)
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("Test decision points (n = %d); shading = row-normalised" % len(y_te), fontsize=9, color=MUTED, x=0.02, ha="left")
fig.tight_layout(); fig.savefig(FIG / "fig4_confusion_matrices.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for name, p in preds.items():
    fpr, tpr, _ = roc_curve(y_te, p); pr, rc, _ = precision_recall_curve(y_te, p)
    a = results["models"][name]["T_0.5"]["all_decision_points"]
    axs[0].plot(fpr, tpr, color=COL[name], lw=2, label=f"{LBL[name]} (AUROC {a['auroc']:.3f})")
    axs[1].plot(rc, pr, color=COL[name], lw=2, label=f"{LBL[name]} (AUPRC {a['auprc']:.3f})")
axs[0].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
axs[1].axhline(y_te.mean(), color=MUTED, ls=":", lw=1); axs[1].text(0.02, y_te.mean() + 0.02, f"prevalence {y_te.mean():.3f}", color=MUTED, fontsize=8)
axs[0].set(xlabel="False-positive rate (1 − specificity)", ylabel="True-positive rate (recall)", title="")
axs[1].set(xlabel="Recall (Y=1)", ylabel="Precision (Y=1)", ylim=(0, 1.02))
axs[0].set_title("ROC, test decision points", fontsize=10, loc="left"); axs[1].set_title("Precision–recall, positive class Y=1", fontsize=10, loc="left")
for ax in axs: ax.grid(True, color=GRID, lw=.8); ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig5_roc_pr.png", dpi=200); plt.close(fig)
print(OUT_MD.read_text(encoding="utf-8"))
