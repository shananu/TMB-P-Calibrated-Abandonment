"""Targeted tests for methodology-critical behaviour not covered elsewhere (added in the final audit).
Run: python -m tests test_methodology"""
import numpy as np

from tmbp.core import DecisionPoint, Episode, crc_threshold, evaluate, rollout
from tmbp.experiments import BetaAttempt, Encoder, bayes_scores, hindsight_scores
from tmbp.generate import build_task, parse_gold


def _dp(calls):
    return DecisionPoint({}, calls, False)


def _toy():
    """A: Y=1, 2 decision points; B: Y=0, 3 dps; C: Y=1 never fails; D: Y=0 never fails."""
    eps = [Episode("A", {}, 1, 3, [_dp(1), _dp(2)]),
           Episode("B", {}, 0, 8, [_dp(1), _dp(2), _dp(3)]),
           Episode("C", {}, 1, 2, []),
           Episode("D", {}, 0, 4, [])]
    scores = [[0.9, 0.05], [0.2, 0.01, 0.01], [], []]
    return eps, scores


def test_evaluate_exact_values_and_denominators():
    eps, sc = _toy()
    r = evaluate(eps, sc, 0.1)  # A stops at its 2nd dp (2 calls), B at its 2nd dp (2 calls)
    assert r["success"] == 0.25 and r["mean_calls"] == 2.5 and r["stop_rate"] == 0.5
    assert r["FAR"] == 0.25                 # joint: 1 abandoned Y=1 episode / 4 episodes
    assert r["FAR_cond"] == 1.0             # denominator = Y=1 episodes WITH >=1 failure (A only; C excluded)
    assert r["waste_reduction"] == 1 - (2 + 4) / (8 + 4)   # Y=0 calls under policy / under never-stop
    r = evaluate(eps, sc, 0.5)  # B now stops at its FIRST dp (score 0.2 < 0.5): first dp below lambda
    assert r["mean_calls"] == (2 + 1 + 2 + 4) / 4 and r["waste_reduction"] == 1 - (1 + 4) / 12
    r = evaluate(eps, sc, 0.0)  # lambda = 0 is never-stop
    assert r["success"] == 0.5 and r["FAR_cond"] == 0 and r["waste_reduction"] == 0
    h = evaluate(eps, hindsight_scores(eps), 0.5)
    assert h["FAR_cond"] == 0 and h["success"] == 0.5 and h["waste_reduction"] == 1 - (1 + 4) / 12


def test_conditional_crc_uses_only_y1_episodes_with_failures_and_is_tight():
    rng = np.random.default_rng(5)
    n = 19
    eps = [Episode(str(i), {}, 1, 5, [_dp(1)]) for i in range(n)]
    sc = [[float(x)] for x in rng.random(n)]
    lam = crc_threshold(eps, sc, 0.1, conditional=True)
    # (k+1)/(n+1) <= 0.1 with n=19 -> at most k=1 calibration episode may be stopped
    assert sum(s[0] < lam for s in sc) == 1
    # adding Y=0 episodes (any scores) and Y=1 episodes without failures must not change lambda
    extra = [Episode("z", {}, 0, 5, [_dp(1)]), Episode("c", {}, 1, 2, [])]
    for z in (0.0, 0.5, 1.0):
        assert crc_threshold(eps + extra, sc + [[z], []], 0.1, conditional=True) == lam
    assert crc_threshold(eps[:5], sc[:5], 0.1, conditional=True) == 0.0  # (0+1)/6 > 0.1 -> never stop


def test_persistent_and_transient_twins_identical_while_fault_fires():
    """Hidden-fault premise: for the same task, persistent and transient faults give identical
    observable features (leak_* excluded) at the first decision point, and at every later decision
    point reached while the transient fault has fired on every call so far (identical error codes).
    NOTE (audit finding): once the transient fault does NOT fire, the twins can differ even without a
    success - e.g. schema drift then returns the environment's genuine 'unknown_argument' error for the
    agent's drift-repaired arguments. That is legitimate observable signal, not leaked metadata."""
    checked = diverged = 0
    for i in range(40):
        tp, tt = build_task("pool", i, pi=1.0, q=0.5, seed=7), build_task("pool", i, pi=0.0, q=0.5, seed=7)
        if parse_gold(tp)["mode"] == "none":
            continue
        assert parse_gold(tp)["mode"] == "persistent" and parse_gold(tt)["mode"] == "transient"
        (ep, rp), (et, rt) = rollout(tp), rollout(tt)
        assert ep.dps and et.dps
        obs = lambda d: {k: v for k, v in d.feats.items() if not k.startswith("leak_")}
        assert obs(ep.dps[0]) == obs(et.dps[0]), f"first failure distinguishable for pool task {i}"
        codes = lambda r: [e["error"]["code"] if e.get("error") else None for e in r.transcript]
        cp, ct = codes(rp), codes(rt)
        for a, b in zip(ep.dps, et.dps):
            if a.calls_used != b.calls_used or cp[:a.calls_used] != ct[:b.calls_used]:
                diverged += 1
                break
            assert obs(a) == obs(b), f"pool task {i}: identical error history but different features"
        checked += 1
    assert checked >= 20 and diverged > 0


def test_encoder_unseen_category_is_all_zero():
    enc = Encoder(["code"], ["consec_fail"]).fit([{"code": "a", "consec_fail": 1}, {"code": "b", "consec_fail": 2}])
    X = enc.transform([{"code": "zzz", "consec_fail": 3}])
    assert X.tolist() == [[3.0, 0.0, 0.0]]


def test_lookup_table_backs_off_to_uniform_prior_on_unseen_key():
    tab = BetaAttempt().fit([])
    ep = Episode("x", {}, 1, 3, [DecisionPoint({"tool": "t", "code": "c", "consec_fail": 9, "budget_total": 8}, 1, False)])
    assert tab.scores([ep]) == [[0.5]]  # Beta(1,1) mean


def test_bayes_reference_formula():
    ep = Episode("x", {}, 1, 5, [DecisionPoint({"consec_fail": 1, "calls_left": 3}, 1, False),
                                 DecisionPoint({"consec_fail": 3, "calls_left": 1}, 3, False)])
    s = bayes_scores([ep], pi=0.5, q=0.5)[0]
    assert np.isclose(s[0], 0.5 * (1 - 0.5 ** 3))                       # P(pers | 1 fail) = pi
    assert np.isclose(s[1], (1 - 0.5 / (0.5 + 0.5 * 0.25)) * (1 - 0.5))  # P(pers | 3 fails) = .5/(.5+.5*.5^2)
