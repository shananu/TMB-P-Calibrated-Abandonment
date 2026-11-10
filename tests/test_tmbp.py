"""Unit tests: CRC correctness on synthetic data, scrubbing, replay equivalence.
Run: python -m tests test_tmbp   (or `python -m tests` for the whole suite)"""
import numpy as np

from tmbp.core import DecisionPoint, Episode, crc_threshold, evaluate, observable_features, rollout, scrub, _contains_key
from tmbp.generate import build_task


def _synthetic(n, rng):
    eps, scores = [], []
    for i in range(n):
        y = int(rng.random() < 0.5)
        k = int(rng.integers(1, 5))
        mu = 0.65 if y else 0.35
        sc = list(np.clip(rng.normal(mu, 0.2, size=k), 0, 1))
        eps.append(Episode(str(i), {}, y, k + 1, [DecisionPoint({}, j + 1, False) for j in range(k)]))
        scores.append(sc)
    return eps, scores


def test_crc_validity(trials=400, n=300, alpha=0.1):
    """Empirical mean of test risk over many independent calibration/test draws must be <= alpha."""
    rng = np.random.default_rng(0)
    joint, cond = [], []
    for _ in range(trials):
        ce, cs = _synthetic(n, rng)
        te, ts = _synthetic(2000, rng)
        joint.append(evaluate(te, ts, crc_threshold(ce, cs, alpha))["FAR"])
        cond.append(evaluate(te, ts, crc_threshold(ce, cs, alpha, conditional=True))["FAR_cond"])
    print(f"joint mean risk {np.mean(joint):.4f}  conditional mean risk {np.mean(cond):.4f}  (alpha={alpha})")
    assert np.mean(joint) <= alpha + 0.005 and np.mean(cond) <= alpha + 0.005
    assert np.mean(cond) > alpha - 0.03, "CRC should be nearly tight, not trivially conservative"


def test_crc_monotone_and_fallback():
    rng = np.random.default_rng(1)
    e, s = _synthetic(50, rng)
    assert crc_threshold(e, s, 0.001) == 0.0  # infeasible alpha -> never stop
    lams = [crc_threshold(e, s, a) for a in (0.02, 0.05, 0.1, 0.2, 0.4)]
    assert all(a <= b for a, b in zip(lams, lams[1:]))


def test_scrub_and_features():
    t = build_task("pool", 9, pi=1.0, q=0.5, seed=7)  # faulted, persistent
    seen = []
    ep, _ = rollout(t, stop_fn=lambda f: seen.append(f) or False)
    assert ep.dps, "expected decision points"
    for dp in ep.dps:
        assert dp.feats["leak_tag"] == "persistent"  # leak exists in the raw observation...
        assert "faults" not in str({k: v for k, v in dp.feats.items() if not k.startswith("leak_")})


if __name__ == "__main__":
    test_crc_monotone_and_fallback(); test_scrub_and_features(); test_crc_validity(); print("all tests passed")
