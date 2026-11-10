"""Tests for the fixed-model / fixed-threshold shift condition.
Run: python -m tests test_shift   (or `python -m tests` for the whole suite)"""
import copy, pickle
import numpy as np
from tmbp.shift import check_aligned, fit_models, freeze, make_splits, run_split, take

def _load(q):
    with open(f"data/tmbp_pi0.5_q{q}/episodes.pkl", "rb") as fh:
        return pickle.load(fh)

def test_splits_disjoint_and_reproducible():
    a, b = make_splits(2160, 5, 1), make_splits(2160, 5, 1)
    for (c1, t1), (c2, t2) in zip(a, b):
        assert np.array_equal(c1, c2) and np.array_equal(t1, t2)
        assert not set(c1) & set(t1) and len(c1) + len(t1) == 2160

def test_calib_and_test_tasks_disjoint_across_q():
    src, tgt = _load(0.5), _load(0.2)
    check_aligned(src["pool"], tgt["pool"])
    ci, ti = make_splits(len(src["pool"]), 1, 3)[0]
    assert not {e.task_id for e in take(src["pool"], ci)} & {e.task_id for e in take(tgt["pool"], ti)}
    assert not {e.task_id for e in src["train"]} & {e.task_id for e in tgt["pool"]}

def test_frozen_threshold_ignores_target_labels_and_model_not_refit():
    src, tgt = _load(0.5), _load(0.8)
    models = fit_models(src["train"], {"0.8": tgt["train"]})
    snap = pickle.dumps(models["src_learned"].model)
    ci, ti = make_splits(len(src["pool"]), 1, 7)[0]
    r1 = run_split(models, src["pool"], {"0.8": tgt["pool"]}, ci, ti, 0.1)
    # corrupt every target label: frozen lambda and frozen predictions must not change
    flipped = copy.deepcopy(tgt["pool"])
    for e in flipped: e.y = 1 - e.y
    r2 = run_split(models, src["pool"], {"0.8": flipped}, ci, ti, 0.1)
    assert r1["lambda_frozen_learned"] == r2["lambda_frozen_learned"]
    assert r1["lambda_frozen_table"] == r2["lambda_frozen_table"]
    assert r1["0.8"]["frozen_learned"]["stop_rate"] == r2["0.8"]["frozen_learned"]["stop_rate"]
    # lambda equals the value computed from source calibration alone
    assert freeze(models["src_learned"], take(src["pool"], ci), 0.1).lam == r1["lambda_frozen_learned"]
    assert pickle.dumps(models["src_learned"].model) == snap, "source model was modified"

def test_cached_scorer_matches_uncached():
    from tmbp.shift import CachedScorer
    src, tgt = _load(0.5), _load(0.2)
    models = fit_models(src["train"], {})
    for k in ("src_learned", "src_table"):
        c = CachedScorer(models[k])
        for pool in (src["pool"][:300], tgt["pool"][:300]):
            a, b = models[k].scores(pool), c.scores(pool)
            assert all(np.allclose(x, y) for x, y in zip(a, b)), k

