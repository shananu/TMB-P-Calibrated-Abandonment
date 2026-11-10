# When Should a Tool-Using Agent Stop Retrying?
An empirical study of calibrated abandonment under hidden persistent faults (TMB-P, built on ToolMisuseBench).

**Framing:** this is an empirical study and a small benchmark extension. It is not a new algorithm. The components (gradient-boosted classifier, conformal risk control, Beta posteriors) are all standard. Closest prior work: Bench2Robust (per-(tool, error) recoverability beliefs, Retry/Switch/Abstain), CodeRescue (cost-penalised router with CRC on *cost*; withdrawn v2), Self-Healing Orchestrators (rule-based escalate/terminate).

## Setup
Python 3.12 (results were produced with 3.12.6 on Windows 11, CPU only). The same commands work on Linux and macOS. No `PYTHONPATH` is needed: every script imports `scripts/_bootstrap.py`, which adds the repository root and the vendored benchmark to `sys.path` and switches to the repository root.
```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
```
ToolMisuseBench is vendored, unmodified, in `third_party/toolmisusebench` at upstream commit `b6ee3412f98d2058ee487e182a72b26c58046d51` (https://github.com/akgitrepos/toolmisusebench, MIT). Byte-identity with that commit was checked with `diff -r` during the final audit.

## Verify the saved artifacts (a few minutes)
```
python scripts/checksums.py verify   # SHA-256 of data/ and results/ vs CHECKSUMS.sha256; exit 1 if any file is changed, missing or unlisted
python -m tests                      # whole test suite (run from the repo root); prints PASS/FAIL per test
```

## Reproduce everything from scratch
Every output script **refuses to overwrite** existing outputs. To regenerate without touching the reference results, write into a separate directory and compare:
```
set TMBP_RESULTS_DIR=results_rerun     # Windows cmd   (PowerShell: $env:TMBP_RESULTS_DIR="results_rerun"; bash: export TMBP_RESULTS_DIR=results_rerun)
```
Run the scripts in this order. Later scripts read earlier outputs.

| step | command | writes | about |
|---|---|---|---|
| 0 | `python scripts/00_reproduce.py` | `data/tmb_v02_large/` (if absent), `00_reproduce.json` | regenerates the seed-42 v0.2 release and reproduces the reference-agent numbers (seconds) |
| 1 | `python scripts/01_verify_benchmark.py` | `01_verify.json` | one-shot faults, label leakage, triggers, replay determinism (seconds) |
| 2 | (called by step 3) `python scripts/02_build_tmbp.py PI Q` | `data/tmbp_piPI_qQ/` | builds one TMB-P variant, rolls out never-stop and checks prefix replay |
| 3 | `python scripts/03_experiments.py` | `E1_main_E7_leakage.json`, `E2_budget.json`, `E3_pi.json`, `E4_q.json`, `E5_heldout_family.json`, `E6_heldout_domain.json` | builds missing variants, then E1–E7 |
| 4 | `python scripts/04_tables.py` | `TABLES.md` | needs step 3 |
| 5 | `python scripts/05_breakdown.py` | `05_breakdown.json` | per-family breakdown |
| 6 | `python scripts/06_figures.py` | `figures/fig1-3*.png` | needs E1, E3, E4 |
| 7 | `python scripts/07_q_shift.py` | `q_shift/` | E8 (about 15 s) |
| 8 | `python scripts/08_classification_metrics.py` | `classification_metrics.json`, `CLASSIFICATION_METRICS.md`, `figures/fig4-5*.png` | needs E1; asserts the refit reproduces the saved AUROC and Brier |

Then `python scripts/checksums.py compare results results_rerun`. `TMBP_OVERWRITE=1` deliberately replaces outputs. The audit's own from-scratch run is described in `FINAL_AUDIT_REPORT.md`.

**Seeds:**
- v0.2 release: seed 42
- TMB-P generation: seed 7
- HistGradientBoosting: `random_state=0`; logistic regression is deterministic
- E1–E7 calibration/test splits: `default_rng(0)`
- breakdown split: 123
- E8 splits: 20261109
- classification bootstrap: 20261110 (+1 for the paired comparison, +2 for the first-decision-point CI)
- `GroupKFold`: deterministic

## Repository layout
```
tmbp/generate.py      TMB-P task builder (fault plans, budgets); ground truth only in gold_summary
tmbp/core.py          rollouts, observation scrubbing, observable features, exact policy evaluation, CRC
tmbp/experiments.py   baselines, learned predictors, E1–E7 protocol (run_policies)
tmbp/shift.py         E8 frozen-calibration protocol
scripts/00–08         pipeline (see table above); _bootstrap.py = path/cwd/overwrite guard; checksums.py
tests/                python -m tests  (test_tmbp, test_shift, test_classification, test_methodology, test_reported_numbers, test_checksums)
data/tmb_v02_large/   regenerated ToolMisuseBench v0.2 release (seed 42)
data/tmbp_pi*_q*/     TMB-P variants: train/pool task JSONL, never-stop episode logs (episodes.pkl), build report
results/              reference outputs (JSON, tables, figures, q_shift/)
third_party/          vendored ToolMisuseBench (unmodified)
CHECKSUMS.sha256      SHA-256 of data/ and results/ (text files hashed with LF line endings)
```
`episodes.pkl` holds pickled `tmbp.core.Episode` objects. Loading needs this repository's `tmbp` package, and a pickle should only be loaded from a trusted source.

## Verified benchmark facts (scripts/00, 01)
| Claim | Evidence |
|---|---|
| Reference numbers reproduce exactly | Parser+Repair 0.7667 task success / 0.7362 conditional recovery; no-recovery 0.1575 (`results/00_reproduce.json`) |
| All released faults are one-shot | 8,076/8,076 fault specs (in the 6,735 faulted tasks among 8,100) use `on_nth_call=1` with no `prob` trigger |
| Recoverability tag = fault family | 1:1 mapping (transient, transient, repairable, transient_denial, transient_with_ambiguous_feedback) |
| Tag leaks to the agent | Present in `last_error.details.faults[].payload.recoverability` after the first faulted call, for 993/993 faulted test tasks |
| Tags ≠ agent-relative recoverability | "repairable" schema drift: Parser+Repair succeeds on only 9/66 tasks per budget profile |
| Persistent and probabilistic faults need no engine change | Omit `on_nth_call` → fires every call. `trigger.prob={p,seed}` → seeded Bernoulli |
| Deterministic replay | 0 mismatches on 600 double runs. Prefix evaluation equals an actual stopped re-run in 2,160/2,160 checks per TMB-P variant (retry-k stop rules, k = 0, 1, 2, on the first 720 pool tasks) |

## TMB-P
- **Base tasks:** ToolMisuseBench v2 scenarios. Train draws on v2 `train` scenarios; the pool draws on v2 `test_public` scenarios.
- **Fault modes:** each faulted task is persistent with probability π, otherwise transient. Transient faults fail the first call, then each later call independently with probability q. The two modes produce identical observations while the fault keeps firing, in particular at the first failure. After a call on which a transient fault does not fire, the agent may succeed or, for schema drift, receive the environment's genuine `unknown_argument` error for its drift-repaired arguments. That is legitimate observable signal, not leaked metadata (`tests/test_methodology.py`).
- **Budgets:** 4 / 8 / 12 calls, with a retry cap that never binds before the call budget.
- **Sizes:** 2,160 train episodes and 2,160 pool episodes. The pool is randomly halved into calibration and test, 50 times.
- **Base agent:** the benchmark's own Parser+Repair, run until it stops or the budget runs out.
- **Features (observable only):**
  - error code and first error code
  - tool, domain
  - consecutive failures, same-code streak, total errors
  - calls used, calls left, retries left, budget total, fraction of budget left
  - whether the arguments changed
  - Excluded: `details.faults`, `fault_plan`, `task_id`, `gold_summary`.
- **Labels:** Y = never-stop outcome. This is the exact counterfactual outcome of "continue".

**Risk definitions:**
- **FAR_cond** = P(stop | Y=1, ≥1 failure). This is the primary metric.
- **FAR (joint)** = P(stop ∧ Y=1). This was the audit's formulation.

Stop rule: stop at the first decision point whose score is below λ. CRC picks the largest candidate λ with (ΣL+1)/(n+1) ≤ α (candidates are just above calibration minimum scores, so the choice is valid but slightly conservative).

**Cost metric:**
- **waste reduction** = 1 − (tool calls spent on Y=0 episodes under the policy) / (tool calls spent on Y=0 episodes under never-stop).
- Calls made before an episode's first failure cannot be avoided, which is why the hindsight oracle reaches 0.861 and not 1.
- Y=1 episodes are excluded from both numerator and denominator, so their cost shows up in FAR_cond and success instead.

**Intervals:**
- **E1–E8 policy tables:** brackets are the 2.5th and 97.5th percentiles over the 50 random calibration/test halvings (20 for E5 and E6). They reflect calibration and test resampling within one pool, not fresh-pool sampling. In E5 and E6 the test set is the whole held-out slice in every split, so only calibration varies.
- **Classification metrics:** episode-cluster bootstrap CIs.
- **Ranking quality** (`waste_at_FARcond`, Fig. 1 frontiers): the threshold is swept on the test pool. This is an optimistic upper bound, not a deployable result.

CRC is validated on synthetic data: mean risk is 0.0997 (joint) and 0.0924 (conditional) for α=0.1.

## Results (α = 0.1 unless noted; full tables in `results/TABLES.md`)
**E1 main (π=0.5, q=0.5):**
| policy | success | calls/episode | FAR_cond | waste reduction |
|---|---|---|---|---|
| never stop | 0.443 | 5.00 | 0 | 0 |
| retry-3 | 0.422 | 3.07 | 0.071 | 0.472 |
| error-type rule (fixed a priori) | 0.344 | 2.25 | 0.340 | 0.641 |
| Beta per (tool, code) + cCRC | 0.417 | 3.62 | 0.091 | 0.331 |
| lookup (tool, code, attempt, budget) + cCRC | 0.418 | 2.85 | 0.087 | 0.524 |
| logistic regression + cCRC | 0.416 | 2.62 | 0.091 | 0.580 |
| **gradient boosting + cCRC** | 0.416 | 2.62 | 0.091 | 0.581 |
| Bayes stop rule (privileged π, q) + cCRC | 0.419 | 3.20 | 0.082 | 0.441 |
| hindsight oracle | 0.443 | 1.54 | 0 | 0.861 |

**Findings:**
1. **Conditional CRC holds α in distribution.** Mean FAR_cond is 0.045, 0.091 and 0.189 for α = 0.05, 0.10 and 0.20, holding across every π, q and budget slice tested. Individual splits do exceed α: the 97.5th percentile is about 0.15 at α=0.1. The guarantee is in expectation only.
2. **The audit's joint-risk formulation is flawed.** It allows abandonment of would-succeed episodes to grow as persistent faults become common: FAR_cond reaches 0.17 at π=0.1, 0.33 at π=0.5 and 0.99 at π=0.9 (Fig. 3). Use class-conditional risk.
3. **The learned policy beats fixed rules and per-(tool, code) beliefs at matched risk, with the gap concentrated in schema drift.**
   - Learned: 0.58 waste reduction at FAR_cond 0.091. Retry-3: 0.47 at 0.071. Beta: 0.33.
   - Almost all of the gain comes from schema drift, which this agent cannot repair: 1.3 calls vs 3.7–4.0 for retry-3. On infrastructure faults it is close to retry-3.
   - Logistic regression equals gradient boosting, and an attempt- and budget-aware lookup table gets most of the way (0.52).
4. **Fixed rules are not robust to fault stickiness.** As q goes from 0.2 to 0.8, retry-3's FAR_cond moves from 0.007 to 0.319 while the calibrated policy stays at about 0.09 (Fig. 2). The cost: at q=0.8 the calibrated policy saves only 36% of wasted calls.
5. **The privileged Bayes stop rule is beaten by an observable-feature model.** It models faults but not agent-side failures.
6. **The guarantee is marginal across families, not per family.**
   - On the main split, FAR_cond is 0.21 for transient timeouts and 0.15 for adversarial errors. The 0.00 on schema drift averages these down.
   - Held-out fault family: timeout breaks the guarantee (0.156). Rate-limit, schema drift, authz and adversarial stay at or below 0.09.
   - Held-out domain: retrieval breaks it (0.178). Crud, files and scheduling stay at or below 0.093.
   - Lookup and Beta baselines fall back to never-stop on unseen keys.
7. **The tight budget (4 calls) leaves little to save.** All calibrated methods reach about 0.30 waste reduction and the learned model has no edge. The gains grow with budget: about 0.30 at 4 calls, 0.52 at 8 and 0.70 at 12.
8. **Leakage ablation.**
   - Adding the visible recoverability tag raises decision-point AUROC from 0.870 to 0.997 and waste reduction from 0.58 to 0.85, against an oracle of 0.86.
   - Adding the visible fault type changes nothing, because it is redundant with the error code.
   - Any evaluation that does not scrub `details.faults` is invalid.
9. **A large gap to the hindsight oracle remains** (0.58 vs 0.86). Persistent and transient faults are indistinguishable until repeated failures accumulate.

## Predictor classification metrics (`scripts/08_classification_metrics.py`)

**What is classified**
- **Unit:** each **decision point**, meaning the observation right after a failed tool call.
- **Label:** Y=1 if letting the base agent continue would end in task success. This is the episode's exact counterfactual outcome, shared by all of its decision points.
- **Model inputs:** observable features only.

**Data**
- **Train:** the π=0.5, q=0.5 train split.
- **Test:** every decision point in all 2,160 pool episodes: 8,660 points, with Y=1 for 1,167 (13.5%). This is the same set behind the AUROC of 0.870 reported earlier. 360 pool episodes never fail, so they are never classified.
- **No per-example predictions were saved earlier.** The models are refit with identical deterministic code on train only. The script asserts that the refit reproduces the saved AUROC and Brier score to within 1e-12.

**Thresholds** (none uses test labels)
- **0.5:** pre-specified.
- **Train-selected:** Youden J on 5-fold out-of-fold predictions on train, with folds grouped by episode.
- **The CRC λ is a stopping threshold, not a classification threshold.** It is reported separately.

**Confidence intervals:** cluster bootstrap over episodes.

**Reportable results (gradient boosting / logistic regression)**

| | GB, threshold 0.5 | LR, threshold 0.5 | GB, train-selected 0.159 | LR, train-selected 0.115 |
|---|---|---|---|---|
| Accuracy | 0.865 | 0.869 | 0.769 | 0.750 |
| Balanced accuracy | 0.554 | 0.549 | 0.799 | 0.807 |
| Precision (Y=1) | 0.495 | 0.567 | 0.351 | 0.337 |
| Recall (Y=1) | 0.129 | 0.112 | 0.840 | 0.884 |
| Specificity | 0.980 | 0.987 | 0.758 | 0.729 |
| F1 (Y=1) / macro / weighted | 0.204 / 0.565 / 0.829 | 0.187 / 0.558 / 0.829 | 0.495 / 0.673 / 0.803 | 0.488 / 0.661 / 0.788 |
| TN / FP / FN / TP | 7340 / 153 / 1017 / 150 | 7393 / 100 / 1036 / 131 | 5683 / 1810 / 187 / 980 | 5461 / 2032 / 135 / 1032 |

Threshold-free metrics:
- AUROC: 0.870 [0.858, 0.881] for gradient boosting, 0.871 [0.859, 0.882] for logistic regression.
- AUPRC: 0.431 vs 0.448.
- Brier score: 0.089 for both.

**Findings**
- **Accuracy at 0.5 mostly reflects class imbalance.** Predicting Y=0 for every point would already score 0.865.
- **The models do not differ.** Every paired difference has a 95% CI that includes 0. For example, ΔAUROC (GB − LR) = −0.001, CI [−0.006, +0.004].
- **The pooled AUROC is inflated by easy repeated negatives.** Restricted to each episode's *first* decision point (one prediction per episode), AUROC is **0.693** [0.670, 0.716] for GB and 0.703 [0.678, 0.727] for LR. At the first failure, persistent and transient faults look identical by construction.

## E8: shift in transient-failure probability without retraining or recalibration (`scripts/07_q_shift.py`, `results/q_shift/`)

**Protocol**
1. **Train.** Fit the gradient-boosted predictor and the (tool, code, attempt, budget) lookup table once, on the q=0.5 *train* episodes (v2 `train` scenarios).
2. **Split.** For each of 50 splits (seed `20261109`), permute the 2,160 pool **indices** once. One half is the calibration indices; the complement is the test indices.
3. **Calibrate and freeze.** Compute the conditional-CRC threshold λ (α=0.1) from the **q=0.5** episodes at the calibration indices only. Freeze the model and λ.
4. **Test.** Evaluate on the **q=0.2** and **q=0.8** episodes at the *test* indices. Target labels are used only for scoring.
5. **Comparisons, on the same split indices:**
   - Retry-3 (needs no calibration).
   - "Recalibrated": the q=0.5 model with λ recomputed on *target-q calibration* indices.
   - "Retrained + recalibrated": a model trained on target-q train data plus λ recomputed on target-q calibration indices. This is the same protocol as E4.

**Leakage controls**
- TMB-P variants share task instances across q: the same `task_id`, family and fault mode, checked by `check_aligned`. Calibration and test are therefore separated by index, so no calibration task appears in any test set.
- The train scenarios are disjoint from the pool.
- `tests/test_shift.py` checks four things:
  - splits are disjoint and reproducible;
  - calibration and test tasks are disjoint across q;
  - flipping every target label leaves the frozen λ and the frozen stop decisions unchanged, and λ equals the value computed from source calibration alone;
  - the model is not refit, and the scoring cache equals uncached scoring.
- Score caching (`CachedScorer`) is a pure speed-up: scores depend only on the frozen model and each episode's own features.
- The configuration, input-pickle MD5s, all 50 split index lists and per-split results are saved in `results/q_shift/`.
- Existing results were not modified; their checksums were verified before and after.

**Results** (mean over 50 splits; mean test set is 1,080 episodes)

| target | policy | FAR_cond [2.5%, 97.5%] | splits > α | success | calls/episode | waste reduction |
|---|---|---|---|---|---|---|
| q=0.2 (Y=1: 489.6; Y=1 with ≥1 failure: 327.4) | frozen learned | 0.028 [0.013, 0.044] | 0% | 0.445 | 2.45 | 0.580 |
| | frozen lookup table | 0.022 [0.010, 0.034] | 0% | 0.447 | 2.67 | 0.525 |
| | retry-3 | 0.008 [0.003, 0.013] | 0% | 0.451 | 2.88 | 0.472 |
| | recalibrated learned | 0.099 [0.060, 0.144] | 50% | 0.423 | 1.98 | 0.696 |
| | retrained + recalibrated | 0.098 [0.063, 0.136] | 42% | 0.424 | 1.96 | 0.701 |
| q=0.8 (Y=1: 405.1; Y=1 with ≥1 failure: 242.9) | frozen learned | **0.324** [0.260, 0.408] | **100%** | 0.302 | 2.84 | 0.580 |
| | frozen lookup table | 0.295 [0.243, 0.384] | 100% | 0.309 | 3.09 | 0.525 |
| | retry-3 | 0.321 [0.286, 0.363] | 100% | 0.303 | 3.31 | 0.474 |
| | recalibrated learned | 0.096 [0.053, 0.161] | 38% | 0.354 | 4.01 | 0.360 |
| | retrained + recalibrated | 0.097 [0.053, 0.163] | 44% | 0.353 | 3.97 | 0.371 |

Never-stop success: 0.453 at q=0.2, 0.375 at q=0.8.

**Findings**
- **The frozen guarantee does not transfer under shift, and the failure is one-sided.**
  - When transient faults become *less* sticky (q=0.2), the frozen policy over-satisfies α (FAR_cond 0.028). It stays safe but gives up savings that recalibration recovers (0.580 → 0.696).
  - When they become *more* sticky (q=0.8), it breaks the guarantee in every split (FAR_cond 0.324, about 3× α). It is then no better than the fixed Retry-3 rule (0.321).
- **Recalibration alone restores α.** Retraining adds almost nothing over recalibrating the q=0.5 model (0.096 vs 0.097 at q=0.8). In this setting the shift is absorbed by the threshold, not by the ranking.
- **Waste reduction for the frozen policies is nearly identical at q=0.2 and q=0.8** (0.580 vs 0.580 for learned, 0.525 vs 0.525 for the table). This is not a bug:
  - Persistent-fault episodes do not depend on q and are identical across the two test sets.
  - They dominate the wasted calls. On split 0, never-stop spends 3,604 of 4,072 failed-episode calls on them at q=0.2, and 3,604 of 4,817 at q=0.8.
  - Per-split values differ (split 0: 0.5909 vs 0.5913).

**Limitations of E8**
- **Only q is shifted.** π, the fault families, the domains and the agent are unchanged, and q is stationary within an episode.
- **Only two target values are tested.**
- **The recalibrated comparisons need labelled target-q calibration episodes** (about 1,080 per split, about 330 of them in the conditional class). This is an upper comparison, not a deployable alternative unless such labels exist.
- **The per-split "> α" rates for calibrated policies (38–50%) are expected.** CRC bounds the *expected* risk, not the risk on each split.
- **Calibration and test halves share the same 2,160-task pool,** so split-to-split variation understates the variation across fresh pools.

## Threats to validity
- **Synthetic faults** with a stationary per-call q.
- **One deterministic base agent.** Its schema-drift weakness drives much of the gain. No LLM agent was evaluated.
- **Train and pool share scenario templates** (different v2 splits and seeds). Measured on TMB-P (π=0.5, q=0.5): 652 of the 2,160 pool instructions occur verbatim in train, and 1,620 share a template once digits and quoted values are masked. The predictors never see instruction text, only tool, error code and counters, so this does not leak features. It does mean pool tasks are not independent of the training tasks. E5 and E6 (held-out family and domain) are the stricter tests. (The ToolMisuseBench paper's instruction-uniqueness ratio of 0.728 is quoted from the paper and was not re-verified here.)
- **The CRC guarantee is marginal and in expectation;** it is not per family and not per split.
- **No absolute cost model:** "waste" counts tool calls only.
- **Bench2Robust's code is unreleased,** so the Beta baseline is our re-implementation of its idea.
- **The error-type rule was fixed before results were seen.** Its authz=0 retries setting is wrong for this benchmark, which models denials as transient.
