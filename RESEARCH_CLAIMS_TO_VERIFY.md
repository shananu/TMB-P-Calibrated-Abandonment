# Research claims that still need human verification

The final engineering audit checks code, data and saved numbers. It cannot check the claims below. Each needs a person to check it before submission. Status **UNVERIFIED** means the audit did not, or could not, confirm it.

## A. Literature and novelty (external sources, not re-checked in the final audit)
| # | Claim (where it appears) | Status | What to check |
|---|---|---|---|
| A1 | Bench2Robust uses per-(tool, error) recoverability beliefs with Retry/Switch/Abstain actions, and its code is unreleased (README framing; `BetaPosterior` docstring) | UNVERIFIED | Read the paper; confirm the description, that no code is released, and the exact citation |
| A2 | CodeRescue is a cost-penalised router with CRC on *cost*, and its v2 was withdrawn (README framing) | UNVERIFIED | Check the arXiv version history and withdrawal notice; cite the right version |
| A3 | "Self-Healing Orchestrators" uses rule-based escalate/terminate (README framing) | UNVERIFIED | Check the paper and citation |
| A4 | No prior work studies calibrated abandonment for tool-using agents in this way (implied by the framing) | UNVERIFIED | Run a fresh literature search close to submission. The novelty audit was a point-in-time search and may be incomplete |
| A5 | The ToolMisuseBench paper reports an instruction-uniqueness ratio of 0.728 (README threats) | UNVERIFIED | Find the number in the paper, or remove it. The repository's own measured overlap (652/2,160 verbatim; 1,620/2,160 by template) is verified |
| A6 | Reproduced reference numbers (Parser+Repair 0.7667 / 0.7362; no-recovery 0.1575) equal the paper's published table | Reproduced from code (verified); match to the paper's table is UNVERIFIED | Compare with the paper's table, including version and split |
| A7 | Conformal risk control citation: Angelopoulos et al., "Conformal Risk Control", Thm 1 (`crc_threshold` docstring) | UNVERIFIED citation details | Cite the right venue/version. The bound implemented is (ΣL+1)/(n+1) ≤ α with loss bounded by 1 |
| A8 | ToolMisuseBench licence is MIT | Verified: `third_party/toolmisusebench/LICENSE` | Confirm that redistributing the vendored copy suits the venue |

## B. Process claims (only the authors can attest)
| # | Claim | What to check |
|---|---|---|
| B1 | The error-type rule (`ERROR_RULE`) was fixed a priori, before any TMB-P result was inspected | No version-control history predates the results, so the repository cannot prove this. State it only if all authors confirm it |
| B2 | Hyperparameters (HGB: 200 iterations, lr 0.05, 15 leaves; LR: C=1) were not tuned on pool/test data | Same as B1: attest or describe as fixed defaults |
| B3 | α values (0.05, 0.1, 0.2) and π/q grids were chosen before results | Attest |

## C. Statistical claims: wording to check in the paper
| # | Correct scope (as supported by the code) | Do not claim |
|---|---|---|
| C1 | Conditional CRC bounds **expected** FAR_cond = P(stop \| Y=1, ≥1 failure). The expectation is over calibration and test draws, assuming exchangeable calibration and test episodes | Bounds on every split (individual splits exceed α; 97.5th percentile ≈ 0.15 at α=0.1), per family, per domain, or under distribution shift |
| C2 | Under q shift with a frozen threshold the guarantee fails (E8: FAR_cond 0.324 at q=0.8). Recalibration on labelled target data restores it | Robustness to shift |
| C3 | Policy-table brackets are percentiles over random calibration/test halvings of one pool | Confidence intervals for a fresh population. Only the classification metrics use episode-cluster bootstrap CIs |
| C4 | `waste_at_FARcond` and the Fig. 1 frontiers sweep the threshold on the test pool, so they are optimistic upper bounds | Deployable performance |
| C5 | Decision-point AUROC 0.870 is pooled over all decision points. At the first failure only it is 0.693 (GB) / 0.703 (LR) | "Distinguishes recoverable failures 87% of the time" |
| C6 | Persistent and transient faults are observationally identical while the fault keeps firing (verified by `tests/test_methodology.py`). After a call on which the transient fault does not fire, legitimate signals appear, such as success or `unknown_argument` under schema drift | "Indistinguishable until the first success" (incorrect for schema drift; corrected in README and `generate.py`) |
| C7 | Results come from one deterministic rule-based agent (Parser+Repair) on synthetic faults | Any claim about LLM agents or real APIs |
| C8 | Learned and LR policies tie: every paired difference CI includes 0 | "Gradient boosting is better" |
