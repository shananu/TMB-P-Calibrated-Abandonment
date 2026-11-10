"""Stop policies (baselines, learned, calibrated) and evaluation protocol for TMB-P."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from tmbp.core import Episode, crc_threshold, evaluate

CAT = ["code", "first_code", "tool", "domain"]
NUM = ["consec_fail", "same_code_streak", "n_errors", "calls_used", "calls_left",
       "retries_left", "budget_total", "frac_left", "args_changed"]
FEATURE_SETS = {
    "observable": (CAT, NUM),
    "leak_fault_type": (CAT + ["leak_fault_type"], NUM),
    "leak_recoverability_tag": (CAT + ["leak_tag"], NUM),
}

# Fixed a priori (written before any TMB-P result was inspected): max retries per error code.
ERROR_RULE = {"timeout": 3, "rate_limited": 3, "ambiguous_error": 2, "authz_denied": 0,
              "missing_required_arg": 1, "unknown_argument": 1, "invalid_argument_type": 1}


class Encoder:
    def __init__(self, cats: list[str], nums: list[str]):
        self.cats, self.nums, self.vocab = cats, nums, {}

    def fit(self, rows: list[dict[str, Any]]) -> "Encoder":
        for c in self.cats:
            self.vocab[c] = sorted({str(r[c]) for r in rows})
        return self

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        X = np.zeros((len(rows), len(self.nums) + sum(len(v) for v in self.vocab.values())))
        for i, r in enumerate(rows):
            X[i, :len(self.nums)] = [float(r[n]) for n in self.nums]
            off = len(self.nums)
            for c in self.cats:
                v = self.vocab[c]
                s = str(r[c])
                if s in v:  # unseen category -> all zeros (held-out family/tool)
                    X[i, off + v.index(s)] = 1.0
                off += len(v)
        return X


def dp_table(eps: list[Episode]) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows, y = [], []
    for e in eps:
        for dp in e.dps:
            rows.append(dp.feats); y.append(e.y)
    return rows, np.array(y)


class Learned:
    def __init__(self, feature_set: str = "observable", model: str = "hgb", seed: int = 0):
        self.enc = Encoder(*FEATURE_SETS[feature_set])
        self.model = (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                     random_state=seed)
                      if model == "hgb" else LogisticRegression(max_iter=2000, C=1.0))

    def fit(self, eps: list[Episode]) -> "Learned":
        rows, y = dp_table(eps)
        X = self.enc.fit(rows).transform(rows)
        if len(set(y)) < 2:
            raise ValueError("training labels are constant")
        self.model.fit(X, y)
        return self

    def scores(self, eps: list[Episode]) -> list[list[float]]:
        out = []
        for e in eps:
            if not e.dps:
                out.append([]); continue
            out.append(list(self.model.predict_proba(self.enc.transform([d.feats for d in e.dps]))[:, 1]))
        return out

    def dp_metrics(self, eps: list[Episode]) -> dict[str, float]:
        rows, y = dp_table(eps)
        if len(set(y)) < 2:
            return {"auroc": float("nan"), "brier": float("nan")}
        p = self.model.predict_proba(self.enc.transform(rows))[:, 1]
        return {"auroc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p))}


class BetaPosterior:
    """Bench2Robust-style recoverability belief per (tool, error code): Beta(1+s, 1+f) over
    the episode-level outcome of training decision points with that key. Not conditioned on
    attempt count or budget. (Our re-implementation; Bench2Robust's code is unreleased.)"""

    def fit(self, eps: list[Episode]) -> "BetaPosterior":
        self.tab: dict[tuple[str, str], list[int]] = {}
        for e in eps:
            for d in e.dps:
                k = (d.feats["tool"], d.feats["code"])
                s, f = self.tab.get(k, [0, 0])
                self.tab[k] = [s + e.y, f + (1 - e.y)]
        return self

    def scores(self, eps: list[Episode]) -> list[list[float]]:
        out = []
        for e in eps:
            row = []
            for d in e.dps:
                s, f = self.tab.get(self.key(d.feats), [0, 0])
                row.append((1 + s) / (2 + s + f))
            out.append(row)
        return out

    def key(self, feats):
        return (feats["tool"], feats["code"])


class BetaAttempt(BetaPosterior):
    """Lookup-table ablation: Beta posterior per (tool, error code, min(consec_fail, 4), budget_total).
    Tests whether a learned model is needed beyond an attempt- and budget-aware table."""

    def fit(self, eps):
        self.tab = {}
        for e in eps:
            for d in e.dps:
                k = self.key(d.feats)
                s, f = self.tab.get(k, [0, 0])
                self.tab[k] = [s + e.y, f + (1 - e.y)]
        return self

    def key(self, feats):
        return (feats["tool"], feats["code"], min(feats["consec_fail"], 4), feats["budget_total"])


def frontier(eps: list[Episode], scores: list[list[float]]) -> list[dict[str, float]]:
    """Threshold sweep (no calibration): FAR_cond vs waste_reduction, for ranking quality."""
    lams = np.unique(np.concatenate([[0.0], np.nextafter(np.array([s for sc in scores for s in sc] or [0.0]), np.inf)]))
    if len(lams) > 400:  # quantile subsample keeps endpoints
        lams = np.unique(np.concatenate([[0.0, lams[-1]], np.quantile(lams, np.linspace(0, 1, 400))]))
    pts = [dict(lam=float(l), **evaluate(eps, scores, l)) for l in lams]
    return [{k: p[k] for k in ("lam", "FAR", "FAR_cond", "waste_reduction", "success", "mean_calls")} for p in pts]


def waste_at_far(front: list[dict[str, float]], target: float) -> float:
    """Best waste reduction achievable with FAR_cond <= target (oracle threshold choice)."""
    ok = [p["waste_reduction"] for p in front if p["FAR_cond"] <= target + 1e-12]
    return max(ok) if ok else 0.0


def rule_scores(eps: list[Episode], rule: Callable[[dict[str, Any]], bool]) -> list[list[float]]:
    """Encode a stop rule as scores: 0 = stop, 1 = continue (use lam = 0.5)."""
    return [[0.0 if rule(d.feats) else 1.0 for d in e.dps] for e in eps]


def bayes_scores(eps: list[Episode], pi: float, q: float) -> list[list[float]]:
    """PRIVILEGED reference: knows the generator's (pi, q). P(recover | n consecutive failures,
    b calls left) = (1 - P(persistent | n)) * (1 - q^b), P(persistent | n) = pi / (pi + (1-pi) q^(n-1)).
    Ignores agent-side failures (parse errors, broken repairs) by design."""
    out = []
    for e in eps:
        row = []
        for d in e.dps:
            n, b = d.feats["consec_fail"], d.feats["calls_left"]
            pp = pi / (pi + (1 - pi) * q ** (n - 1))
            row.append((1 - pp) * (1 - q ** b))
        out.append(row)
    return out


def hindsight_scores(eps: list[Episode]) -> list[list[float]]:
    return [[float(e.y)] * len(e.dps) for e in eps]


def take(xs, idx):
    return [xs[i] for i in idx]


def run_policies(train: list[Episode], pool: list[Episode], alphas=(0.05, 0.1, 0.2), n_splits: int = 50,
                 pi: float | None = None, q: float | None = None, feature_sets=("observable",),
                 calib_pool: list[Episode] | None = None, seed: int = 0) -> dict[str, Any]:
    """If calib_pool is None: pool is randomly halved into calibration/test, n_splits times
    (exchangeable setting). Otherwise calibration uses calib_pool halves and test = pool (shift setting)."""
    rng = np.random.default_rng(seed)
    beta = BetaPosterior().fit(train)
    beta_att = BetaAttempt().fit(train)
    learned = {fs: Learned(fs).fit(train) for fs in feature_sets}
    lr = Learned("observable", model="lr").fit(train)

    def all_scores(eps):
        s = {
            "never_stop": [[1.0] * len(e.dps) for e in eps],
            "stop_first_failure": rule_scores(eps, lambda f: True),
            "retry_1": rule_scores(eps, lambda f: f["consec_fail"] > 1),
            "retry_2": rule_scores(eps, lambda f: f["consec_fail"] > 2),
            "retry_3": rule_scores(eps, lambda f: f["consec_fail"] > 3),
            "error_type_rule": rule_scores(eps, lambda f: f["consec_fail"] > ERROR_RULE.get(f["code"], 1)),
            "beta_tool_code": beta.scores(eps),
            "table_tool_code_attempt_budget": beta_att.scores(eps),
            "learned_lr": lr.scores(eps),
            "hindsight_oracle": hindsight_scores(eps),
        }
        for fs, m in learned.items():
            s["learned_hgb" + ("" if fs == "observable" else f"[{fs}]")] = m.scores(eps)
        if pi is not None:
            s["bayes_privileged"] = bayes_scores(eps, pi, q)
        return s

    fixed = ["never_stop", "stop_first_failure", "retry_1", "retry_2", "retry_3", "error_type_rule", "hindsight_oracle"]
    thresholded = ["beta_tool_code", "table_tool_code_attempt_budget", "learned_lr"] +["learned_hgb" + ("" if fs == "observable" else f"[{fs}]") for fs in feature_sets]
    if pi is not None:
        thresholded.append("bayes_privileged")

    S_pool = all_scores(pool)
    S_cal = all_scores(calib_pool) if calib_pool is not None else None
    records: dict[str, list[dict[str, float]]] = {}
    for _ in range(n_splits):
        if calib_pool is None:
            perm = rng.permutation(len(pool)); h = len(pool) // 2
            ci, ti = perm[:h], perm[h:]
            cal_eps, test_eps = take(pool, ci), take(pool, ti)
            cal_sc = {k: take(v, ci) for k, v in S_pool.items()}
            test_sc = {k: take(v, ti) for k, v in S_pool.items()}
        else:
            perm = rng.permutation(len(calib_pool)); ci = perm[: len(calib_pool) // 2]
            cal_eps = take(calib_pool, ci); cal_sc = {k: take(v, ci) for k, v in S_cal.items()}
            test_eps, test_sc = pool, S_pool
        for name in fixed:
            records.setdefault(name, []).append(evaluate(test_eps, test_sc[name], 0.5))
        for name in thresholded:
            records.setdefault(f"{name}@0.5", []).append(evaluate(test_eps, test_sc[name], 0.5))
            for a in alphas:
                lam = crc_threshold(cal_eps, cal_sc[name], a)
                r = evaluate(test_eps, test_sc[name], lam); r["lambda"] = lam
                records.setdefault(f"{name}+CRC(a={a})", []).append(r)
                lam = crc_threshold(cal_eps, cal_sc[name], a, conditional=True)
                r = evaluate(test_eps, test_sc[name], lam); r["lambda"] = lam
                records.setdefault(f"{name}+cCRC(a={a})", []).append(r)
    summary = {}
    for name, rs in records.items():
        df = pd.DataFrame(rs)
        summary[name] = {c: {"mean": float(df[c].mean()), "lo": float(df[c].quantile(0.025)),
                             "hi": float(df[c].quantile(0.975))} for c in df.columns}
    diag = {fs: m.dp_metrics(pool) for fs, m in learned.items()}
    diag["learned_lr"] = lr.dp_metrics(pool)
    # ranking quality on the full test pool: best achievable waste reduction at FAR_cond <= t
    fronts = {k: frontier(pool, S_pool[k]) for k in thresholded + ["hindsight_oracle"]}
    rules = {k: evaluate(pool, S_pool[k], 0.5) for k in fixed}
    waste_at = {k: {str(t): waste_at_far(f, t) for t in (0.02, 0.05, 0.1, 0.2)} for k, f in fronts.items()}
    for k, r in rules.items():  # rule baselines: single operating point
        waste_at[k] = {str(t): (r["waste_reduction"] if r["FAR_cond"] <= t + 1e-12 else None) for t in (0.02, 0.05, 0.1, 0.2)}
    return {"policies": summary, "predictor_diagnostics_on_test_pool": diag, "frontiers": fronts,
            "rule_points": rules, "waste_at_FARcond": waste_at}
