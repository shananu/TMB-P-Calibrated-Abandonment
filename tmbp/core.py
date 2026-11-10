"""Rollouts, observable-only features, stop policies and conformal risk control for TMB-P."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from toolmisusebench.baselines import build_agent
from toolmisusebench.generator.core import TOOLSET_BY_DOMAIN
from toolmisusebench.harness.runner import make_env_for_task, run_episode
from toolmisusebench.types import Observation, Task

from tmbp.generate import parse_gold

TOOL_TO_DOMAIN = {t["name"]: dom for dom, (_, schemas) in TOOLSET_BY_DOMAIN.items() for t in schemas}

# --------------------------------------------------------------------------- scrubbing
LEAKY_KEYS = ("faults",)


def scrub(obs: Observation) -> Observation:
    """Remove every benchmark-injected fault annotation from an observation.

    ToolMisuseBench writes FaultEvent.__dict__ (incl. payload.recoverability and fault_type)
    into error.details['faults'] of both `last_error` and transcript entries. Nothing else
    in the observation carries ground truth (task_id/gold_summary/fault_plan are never
    exposed to the agent)."""
    o = obs.model_copy(deep=True)
    if o.last_error is not None:
        for k in LEAKY_KEYS:
            o.last_error.details.pop(k, None)
    for e in o.transcript:
        if e.get("error"):
            for k in LEAKY_KEYS:
                e["error"].get("details", {}).pop(k, None)
        if isinstance(e.get("output"), dict):
            for k in LEAKY_KEYS:
                e["output"].pop(k, None)
    return o


def _contains_key(x: Any, key: str) -> bool:
    if isinstance(x, dict):
        return key in x or any(_contains_key(v, key) for v in x.values())
    if isinstance(x, list):
        return any(_contains_key(v, key) for v in x)
    return False


def observable_features(raw: Observation) -> dict[str, Any]:
    """Features computed ONLY from the scrubbed observation (+ leak-only extras, prefixed
    'leak_', computed from the raw observation for the leakage ablation)."""
    obs = scrub(raw)
    assert not _contains_key(obs.model_dump(), "faults"), "scrub failed"
    tr = obs.transcript
    last = tr[-1]
    codes = [e["error"]["code"] if e.get("error") else None for e in tr]
    consec = 0
    for c in reversed(codes):
        if c is None:
            break
        consec += 1
    same_code = 0
    for c in reversed(codes):
        if c != codes[-1]:
            break
        same_code += 1
    used = len(tr)
    left = obs.remaining_budget.tool_calls_left
    f = {
        "code": codes[-1],
        "first_code": next(c for c in codes if c is not None),
        "tool": last["tool_name"],
        "domain": TOOL_TO_DOMAIN.get(last["tool_name"], "unknown"),
        "consec_fail": consec,
        "same_code_streak": same_code,
        "n_errors": sum(c is not None for c in codes),
        "calls_used": used,
        "calls_left": left,
        "retries_left": obs.remaining_budget.retries_left,
        "budget_total": used + left,
        "frac_left": left / max(1, used + left),
        "args_changed": int(len(tr) >= 2 and tr[-1].get("args") != tr[-2].get("args")),
    }
    # ---- leak-only fields (never used except in the leakage ablation) ----
    faults = (raw.last_error.details.get("faults", []) if raw.last_error else [])
    f["leak_fault_type"] = "|".join(sorted({x.get("fault_type", "") for x in faults})) or "none"
    tags = sorted({x.get("payload", {}).get("recoverability") for x in faults if x.get("payload", {}).get("recoverability")})
    f["leak_tag"] = "|".join(tags) or "none"
    return f


# --------------------------------------------------------------------------- rollouts
@dataclass
class DecisionPoint:
    feats: dict[str, Any]
    calls_used: int           # tool calls consumed if we STOP here
    success_if_stop: bool     # env.check_success() at this moment (should be False)


@dataclass
class Episode:
    task_id: str
    gold: dict[str, str]
    y: int                    # never-stop outcome == counterfactual "continue" outcome
    calls_never_stop: int
    dps: list[DecisionPoint] = field(default_factory=list)


class _Wrapper:
    """Wraps the reference agent. At every decision point (observation with last_error)
    it logs observable features and asks `stop_fn` whether to abandon the episode."""

    def __init__(self, env, stop_fn: Callable[[dict[str, Any]], bool] | None = None, base: str = "instruction_parser"):
        self.base = build_agent(base)
        self.env = env
        self.stop_fn = stop_fn
        self.dps: list[DecisionPoint] = []

    def reset(self) -> None:
        self.base.reset()
        self.dps = []

    def act(self, obs: Observation):
        if obs.last_error is not None and obs.transcript:
            feats = observable_features(obs)
            self.dps.append(DecisionPoint(feats, len(obs.transcript), self.env.check_success().success))
            if self.stop_fn is not None and self.stop_fn(feats):
                return None
        return self.base.act(obs)


def rollout(task: Task, stop_fn=None) -> tuple[Episode, Any]:
    env = make_env_for_task(task)
    agent = _Wrapper(env, stop_fn)
    res = run_episode(task, agent, env=env)
    ep = Episode(task.task_id, parse_gold(task), int(res.success), res.tool_calls, agent.dps)
    return ep, res


# --------------------------------------------------------------------------- policy evaluation (exact, from never-stop logs)
def stop_index(scores: list[float], lam: float) -> int | None:
    for j, s in enumerate(scores):
        if s < lam:
            return j
    return None


def evaluate(episodes: list[Episode], scores: list[list[float]], lam: float) -> dict[str, float]:
    """Outcome of 'stop at the first decision point whose score < lam'. Exact because the
    environment and reference agent are deterministic, so a stopped trajectory is a prefix
    of the never-stop trajectory (verified empirically in scripts/02_build_tmbp.py)."""
    n = len(episodes)
    succ = calls = stop = far = waste_ns = waste = 0
    n_y1_dp = 0
    for ep, sc in zip(episodes, scores):
        j = stop_index(sc, lam)
        if j is None:
            s, c = ep.y, ep.calls_never_stop
        else:
            s, c = int(ep.dps[j].success_if_stop), ep.dps[j].calls_used
            stop += 1
            far += ep.y            # abandoned an episode that would have succeeded
        succ += s
        calls += c
        if ep.y == 0:
            waste_ns += ep.calls_never_stop
            waste += c
        if ep.y == 1 and ep.dps:
            n_y1_dp += 1
    return {
        "success": succ / n,
        "mean_calls": calls / n,
        "stop_rate": stop / n,
        "FAR": far / n,                                   # CRC-controlled (joint) risk
        "FAR_cond": far / max(1, n_y1_dp),                # among recoverable episodes that hit a failure
        "wasted_calls": waste / n,                        # calls spent on episodes whose never-stop outcome is Y=0
        "waste_reduction": 1 - waste / max(1, waste_ns),  # 1 - (calls on Y=0 episodes under policy) / (calls on Y=0 episodes under never-stop)
    }


# --------------------------------------------------------------------------- conformal risk control
def crc_threshold(episodes: list[Episode], scores: list[list[float]], alpha: float,
                  conditional: bool = False) -> float:
    """Conformal risk control for the stop threshold.

    joint (conditional=False): L_i(lam) = 1[min_j s_ij < lam] * Y_i over ALL calibration episodes.
        Guarantee: E[L_test] <= alpha, i.e. P(stop AND would-have-succeeded) <= alpha.
    conditional (conditional=True): restrict calibration to episodes with Y_i = 1 that reach >= 1
        decision point; L_i(lam) = 1[min_j s_ij < lam].
        Guarantee: P(stop | Y = 1, a failure occurred) <= alpha (class-conditional / Mondrian CRC).

    Both losses are non-decreasing in lam and bounded by 1, so
    lam_hat = max{lam : (sum_i L_i(lam) + 1)/(n+1) <= alpha}, with lam = 0 (never stop, L = 0)
    as the fallback, satisfies the CRC bound (Angelopoulos et al., 2022, Thm 1, applied to -lam)."""
    if conditional:
        keep = [i for i, e in enumerate(episodes) if e.y == 1 and e.dps]
        episodes = [episodes[i] for i in keep]
        scores = [scores[i] for i in keep]
    n = len(episodes)
    if n == 0:
        return 0.0
    mins = np.array([min(s) if s else np.inf for s in scores])
    y = np.array([e.y for e in episodes])
    cands = np.unique(np.concatenate([[0.0], np.nextafter(mins[np.isfinite(mins)], np.inf)]))
    best = 0.0
    for lam in cands:  # ascending; risk is monotone so we can stop at first violation
        risk_sum = float(((mins < lam) & (y == 1)).sum())
        if (risk_sum + 1.0) / (n + 1.0) <= alpha:
            best = float(lam)
        else:
            break
    return best
