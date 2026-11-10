"""Distribution-shift protocol: train + calibrate at a source q, freeze, evaluate at target q.

Leakage controls
  * Model fitted once on SOURCE train episodes (v2 'train' scenarios, disjoint from the pool).
  * Per split, pool INDICES are permuted once: calibration uses source-q episodes at calib_idx,
    testing uses target-q episodes at the complementary test_idx. Because TMB-P variants share
    task instances across q (same task_id, family, mode), index-disjointness is what makes the
    calibration and test tasks independent.
  * The frozen threshold is computed from source calibration episodes only, before target data
    is touched; target labels are used only to score the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tmbp.core import Episode, crc_threshold, evaluate
from tmbp.experiments import BetaAttempt, Learned, rule_scores


def take(xs, idx):
    return [xs[int(i)] for i in idx]


def check_aligned(a: list[Episode], b: list[Episode]) -> None:
    assert len(a) == len(b), "pools differ in size"
    for x, y in zip(a, b):
        assert x.task_id == y.task_id and x.gold["mode"] == y.gold["mode"] and x.gold["family"] == y.gold["family"], \
            f"pool misaligned at {x.task_id}"


def make_splits(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        h = n // 2
        out.append((np.sort(perm[:h]), np.sort(perm[h:])))
    return out


@dataclass
class FrozenPolicy:
    """A fitted scorer plus a threshold computed once from source calibration data."""
    scorer: Any
    lam: float

    def evaluate(self, test_eps: list[Episode]) -> dict[str, float]:
        return evaluate(test_eps, self.scorer.scores(test_eps), self.lam)


def freeze(scorer, calib_eps: list[Episode], alpha: float) -> FrozenPolicy:
    return FrozenPolicy(scorer, crc_threshold(calib_eps, scorer.scores(calib_eps), alpha, conditional=True))


def counts(test_eps: list[Episode]) -> dict[str, int]:
    return {"n_test": len(test_eps),
            "n_Y1": int(sum(e.y for e in test_eps)),                       # counterfactually successful
            "n_Y1_with_failure": int(sum(e.y == 1 and bool(e.dps) for e in test_eps))}  # FAR_cond denominator


def run_split(models: dict[str, Any], src_pool: list[Episode], tgt_pools: dict[str, list[Episode]],
              calib_idx: np.ndarray, test_idx: np.ndarray, alpha: float) -> dict[str, Any]:
    """models: {'src_learned', 'src_table', 'tgt_learned:<q>'} already fitted (never refit here)."""
    assert not set(calib_idx.tolist()) & set(test_idx.tolist()), "calib/test indices overlap"
    src_cal = take(src_pool, calib_idx)
    # ---- frozen: thresholds from SOURCE calibration only, computed before any target access ----
    frozen_learned = freeze(models["src_learned"], src_cal, alpha)
    frozen_table = freeze(models["src_table"], src_cal, alpha)
    res: dict[str, Any] = {"lambda_frozen_learned": frozen_learned.lam, "lambda_frozen_table": frozen_table.lam}
    for q, tgt_pool in tgt_pools.items():
        test = take(tgt_pool, test_idx)
        tgt_cal = take(tgt_pool, calib_idx)
        r = {"counts": counts(test)}
        r["frozen_learned"] = frozen_learned.evaluate(test)
        r["frozen_table"] = frozen_table.evaluate(test)
        r["retry_3"] = evaluate(test, rule_scores(test, lambda f: f["consec_fail"] > 3), 0.5)
        r["never_stop"] = evaluate(test, [[1.0] * len(e.dps) for e in test], 0.5)
        # upper comparisons (use target-q CALIBRATION indices; never test indices)
        rc = freeze(models["src_learned"], tgt_cal, alpha)
        r["recalibrated_learned"] = {**rc.evaluate(test), "lambda": rc.lam}
        rt = freeze(models[f"tgt_learned:{q}"], tgt_cal, alpha)
        r["retrained_recalibrated_learned"] = {**rt.evaluate(test), "lambda": rt.lam}
        res[q] = r
    return res


def fit_models(src_train: list[Episode], tgt_trains: dict[str, list[Episode]]) -> dict[str, Any]:
    m = {"src_learned": Learned().fit(src_train), "src_table": BetaAttempt().fit(src_train)}
    for q, tr in tgt_trains.items():
        m[f"tgt_learned:{q}"] = Learned().fit(tr)
    return m


class CachedScorer:
    """Performance wrapper: memoises a FITTED scorer's per-episode scores, keyed by episode object
    identity (pools for different q are distinct objects even though task_ids coincide).
    Scores depend only on the frozen model and the episode's own features - never on labels or
    on the split - so caching cannot leak information. Learned models are scored in one batch."""

    def __init__(self, scorer):
        self.scorer, self._cache = scorer, {}

    def prime(self, eps: list[Episode]) -> None:
        todo = [e for e in eps if id(e) not in self._cache]
        if isinstance(self.scorer, Learned):
            rows = [d.feats for e in todo for d in e.dps]
            p = self.scorer.model.predict_proba(self.scorer.enc.transform(rows))[:, 1] if rows else []
            k = 0
            for e in todo:
                self._cache[id(e)] = [float(x) for x in p[k:k + len(e.dps)]]; k += len(e.dps)
        else:
            for e, s in zip(todo, self.scorer.scores(todo)):
                self._cache[id(e)] = s

    def scores(self, eps: list[Episode]) -> list[list[float]]:
        self.prime(eps)
        return [self._cache[id(e)] for e in eps]
