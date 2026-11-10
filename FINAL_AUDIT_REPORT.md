# Final audit report: TMB-P (calibrated abandonment under hidden persistent faults)

Audit date: 2026-11-10. Environment: Python 3.12.6, Windows 11, CPU only. Package versions are in `requirements.txt`.

## 1. Repository inventory
| path | contents |
|---|---|
| `tmbp/` | `generate.py` (TMB-P builder), `core.py` (rollouts, scrubbing, features, exact evaluation, CRC), `experiments.py` (baselines, predictors, E1–E7), `shift.py` (E8) |
| `scripts/` | `00`–`08` pipeline; `_bootstrap.py` (paths, working directory, overwrite guard; new); `checksums.py` (new) |
| `tests/` | `test_tmbp`, `test_shift`, `test_classification`, `test_methodology` (new), `test_reported_numbers` (new); `__main__.py` runner (new) |
| `data/` | `tmb_v02_large/` (regenerated v0.2 release); 7 TMB-P variants `tmbp_pi*_q*/` (task JSONL, `episodes.pkl`, build report) |
| `results/` | E1–E6 JSON, `00_reproduce.json`, `01_verify.json`, `05_breakdown.json`, `TABLES.md`, `q_shift/` (E8), classification metrics JSON/MD, `figures/fig1–fig5` |
| `third_party/toolmisusebench` | vendored benchmark, 86 files |
| new top-level files | `requirements.txt`, `.python-version`, `.gitignore`, `.gitattributes`, `CHECKSUMS.sha256`, `RESEARCH_CLAIMS_TO_VERIFY.md`, this report |

All classification-metrics artifacts are present: script 08, its JSON and MD outputs, fig4, fig5 and `tests/test_classification.py`. No file is duplicated or obsolete. The largest file is 8.4 MB, the repository has no secrets or API keys, and no machine-specific paths appear in code. Personal information is limited to platform strings in `results/q_shift/config.json`.

## 2. Issues discovered
| # | severity | issue |
|---|---|---|
| I1 | result text | README stated "8,100/8,100 fault specs are one-shot". `01_verify.json` shows **8,076** fault specs (in 6,735 faulted tasks among 8,100). All of them are one-shot, so the conclusion is unchanged |
| I2 | result text | README quoted standard-budget waste reduction as 0.53; the saved value is 0.5247, which is **0.52** |
| I3 | claim | "Persistent and transient faults produce identical errors until the first success" is false for schema drift. After a call on which the transient fault does not fire, the environment returns a genuine `unknown_argument` error for the agent's drift-repaired arguments. This is legitimate observable signal, not leakage. Identity holds while the fault keeps firing, which includes the first failure (verified by a new test) |
| I4 | reproducibility | Scripts 00–06 silently overwrote reference outputs. Scripts 05, 06 and 08 assumed `results/` and `results/figures/` already existed. Every command required a `PYTHONPATH` whose separator differs between Windows and POSIX |
| I5 | documentation | The generator docstring referenced a non-existent `tmbp/features.py`. The comment on `wasted_calls` misdefined it as "episodes that end in failure"; the correct definition is "episodes whose never-stop outcome is Y=0" |
| I6 | documentation | The README did not define the waste-reduction denominator or say what the brackets are. Policy-table brackets are percentiles over random splits, not bootstrap CIs. Frontier and `waste_at_FARcond` values use thresholds swept on the test pool, so they are optimistic upper bounds |
| I7 | code quality | `tests/test_shift.py` had two `__main__` blocks. Pickle files were opened without being closed |
| I8 | limitation (measured) | 652/2,160 pool instructions appear verbatim in train, and 1,620/2,160 match a train template once digits and quoted values are masked. Features never include text, so there is no feature leakage, but tasks are not independent |
| I9 | minor | The `retries_left` feature always equals `calls_left`. It is redundant but harmless and was left unchanged, because changing it would change results |
| I10 | missing | No dependency pins, Python version, `.gitignore`, checksums or test runner existed |

**Checked and found correct:**
- **Scrubbing and features:** `details.faults` is scrubbed before feature extraction, and the observable feature set has no `leak_*`, `task_id`, `gold_summary` or `fault_plan` fields.
- **Labels:** labels are the exact never-stop outcome. Prefix replay matched in 2,160/2,160 checks per variant, and no decision point has already-successful tasks.
- **Data separation:**
  - train uses v2 `train` scenarios, which are disjoint from the pool's v2 `test_public` scenarios;
  - calibration and test halves are split by episode;
  - the GroupKFold threshold is grouped by episode;
  - E8 separates calibration from test by index.
- **Conditional CRC:**
  - calibrates only on Y=1 episodes with at least one failure;
  - uses the bound (ΣL+1)/(n+1) ≤ α;
  - falls back to λ=0;
  - is tight on synthetic data (mean risk 0.0924 at α=0.1).
- **FAR_cond denominator:** Y=1 episodes with at least one failure, so episodes without a failure are excluded.
- **Budget features:** consistent at all 17,235 decision points (calls used + calls left = budget, frac_left exact).
- **Logistic regression:** converges in 108 iterations.
- **Seeds:** all fixed.
- **Thresholds:** no test labels are used for fitting or threshold selection anywhere.

## 3. Fixes made (no methodology or numeric results changed)
- `scripts/_bootstrap.py`:
  - sets paths, makes the repository root the working directory, and forces UTF-8 output;
  - adds the `TMBP_RESULTS_DIR` output redirect;
  - refuses to overwrite outputs unless `TMBP_OVERWRITE=1`;
  - creates output directories (I4).
- Every script imports it; pickle files are now opened with context managers (I4, I7).
- README corrections for I1, I2, I3, I6 and I8, plus setup steps, the run-order table, seeds, a layout section and pickle-trust advice.
- `tmbp/generate.py` docstring corrections (I3, I5) and the `tmbp/core.py` comment fix (I5).
- `tests/__init__.py` and `tests/__main__.py` let `python -m tests` run on any OS. The duplicate `__main__` block in `test_shift` was removed.
- New tests:
  - `test_methodology.py`: exact metric values and denominators, CRC invariance to Y=0 and no-failure episodes and its tightness, persistent/transient twin identity, encoder behaviour on unseen categories, lookup-table prior, Bayes formula.
  - `test_reported_numbers.py`: every README headline number against saved artifacts.
- `requirements.txt`, `.python-version`, `.gitignore`, `.gitattributes`, `scripts/checksums.py`, `CHECKSUMS.sha256`.

## 4. Tests executed
`python -m tests` on the final code gave **20/20 passed** (exit 0). Log: `_audit_repro/tests_main.txt`, git-ignored.

| module | tests | result |
|---|---|---|
| test_tmbp | 3 (CRC validity: joint 0.0997, conditional 0.0924 at α=0.1; monotonicity and fallback; scrubbing) | pass |
| test_shift | 4 | pass |
| test_classification | 3 | pass |
| test_methodology (new) | 6 | pass |
| test_reported_numbers (new) | 4 | pass |

Bug found by writing the new tests: the first version of the twin test failed, which exposed I3.

## 5. Experiments reproduced
- `python scripts/checksums.py verify`: 62/62 data and results files match the manifest. Reference outputs were also confirmed byte-identical to a pre-audit MD5 snapshot.
- **Vendored ToolMisuseBench:** a fresh clone of upstream commit `b6ee3412f98d2058ee487e182a72b26c58046d51`, compared with `diff -r`, is **identical**.
- **From scratch, in an isolated copy with no `data/` or `results/`:**
  - `00_reproduce.py` regenerated `data/tmb_v02_large` **identical** to the saved copy (5/5 files) and reproduced 0.7667 / 0.7362 / 0.1575.
  - `01_verify_benchmark.py` ran successfully.
  - The from-scratch run was then **stopped by the user** during TMB-P rebuilding, so rebuilding the TMB-P data was **not re-verified** in this audit. Each variant's build report records 0 replay mismatches from the original build.
- **Re-run of 03–08 on the saved TMB-P data** into `_audit_repro/results_rerun`, compared with `checksums.py compare`: see section 6.

## 6. Result discrepancies
- No saved numeric result was changed. The only discrepancies were README text (I1, I2, I3); `tests/test_reported_numbers.py` now guards against recurrence.
- Re-run comparison of 03–08: the re-run was **stopped by the system for low memory** partway through script 03. It completed `E1_main_E7_leakage.json` and `E2_budget.json`, and both are **byte-identical** to the saved results. E3–E6 and scripts 04–08 were **not re-run** (UNVERIFIED by re-execution). They remain covered by the checksum manifest, `test_reported_numbers`, and script 08's built-in assertion that it reproduces E1's AUROC/Brier.

## 7. Missing artifacts
- No per-example predictions were saved by E1–E7. Script 08 refits deterministically and asserts it reproduces the saved AUROC/Brier to within 1e-12.
- Per-split raw records for E1–E7 are not saved; only summaries and frontiers are.
- E1–E7 store no environment metadata (versions and runtimes); E8 does.

## 8. Remaining limitations
- One deterministic rule-based agent and synthetic faults with a stationary q; no LLM agent was evaluated.
- The CRC guarantee is in expectation and marginal: individual splits exceed α, and it fails for held-out family (timeout 0.156), held-out domain (retrieval 0.178) and frozen q shift (0.324 at q=0.8).
- Policy brackets come from resampling one pool, not from fresh data. Frontier figures are optimistic.
- Train/pool template overlap (I8).
- Pickled episode logs depend on `tmbp` class definitions and pinned library versions. PNG and pickle byte-identity is only expected with `requirements.txt`.
- Not re-verified in this audit: the from-scratch TMB-P rebuild (step 02) and a Linux run.

## 9. Git
Local repository initialised on branch `main`. Initial commit `50906ac2d5ddc2bade4244784198375b95dcfeb7` contains 179 files (code, tests, vendored benchmark, data, results). This report's Git section was added in a follow-up commit. No remote was created and nothing was pushed. Git normalises text files to LF line endings in the repository (`.gitattributes`); `CHECKSUMS.sha256` hashes text with LF, so `checksums.py verify` works on any checkout. `_audit_repro/` (partial re-run outputs and logs) is git-ignored.

## 10. Reproduction commands
```
python -m pip install -r requirements.txt
python scripts/checksums.py verify
python -m tests
# full regeneration into a separate directory (bash syntax; see README for Windows):
export TMBP_RESULTS_DIR=results_rerun
python scripts/00_reproduce.py && python scripts/01_verify_benchmark.py && python scripts/03_experiments.py \
 && python scripts/04_tables.py && python scripts/05_breakdown.py && python scripts/06_figures.py \
 && python scripts/07_q_shift.py && python scripts/08_classification_metrics.py
python scripts/checksums.py compare results results_rerun
```
Claims that need human verification are listed in `RESEARCH_CLAIMS_TO_VERIFY.md`.
