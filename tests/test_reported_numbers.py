"""Every headline number quoted in README.md, checked against the saved result artifacts.
If an experiment is re-run and a number changes, this test fails and the README must be updated.
Run: python -m tests test_reported_numbers"""
import json
from pathlib import Path

R = Path("results")


def _j(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def _m(pol, name, metric):
    return pol[name][metric]["mean"]


def _close(got, want, tol, what):
    """want is the README value; tol is half a unit in its last printed digit (rounding tolerance)."""
    assert abs(got - want) <= tol + 1e-9, f"{what}: saved {got:.4f} vs README {want}"


def test_benchmark_facts():
    r = _j("00_reproduce.json")
    _close(r["instruction_parser"]["task_success"], 0.7667, 5e-5, "Parser+Repair success")
    _close(r["instruction_parser"]["cond_recovery"], 0.7362, 5e-5, "Parser+Repair cond. recovery")
    _close(r["instruction_parser_no_recovery"]["task_success"], 0.1575, 5e-5, "no-recovery success")
    v = _j("01_verify.json")
    trig = v["trigger_counts(fault_type,on_nth_call,has_prob)"]
    assert v["n_tasks"] == 8100 and sum(trig.values()) == 8076
    assert all("'1', False" in k for k in trig), "a released fault spec is not one-shot"
    assert sum(v["observed_after_first_call(family,visible_recoverability,visible_fault_type)"].values()) == 993
    assert v["replay_mismatches_first600"] == 0


def test_E1_main_table():
    p = _j("E1_main_E7_leakage.json")["policies"]
    rows = {"never_stop": (0.443, 5.00, 0.0, 0.0), "retry_3": (0.422, 3.07, 0.071, 0.472),
            "error_type_rule": (0.344, 2.25, 0.340, 0.641),
            "beta_tool_code+cCRC(a=0.1)": (0.417, 3.62, 0.091, 0.331),
            "table_tool_code_attempt_budget+cCRC(a=0.1)": (0.418, 2.85, 0.087, 0.524),
            "learned_lr+cCRC(a=0.1)": (0.416, 2.62, 0.091, 0.580),
            "learned_hgb+cCRC(a=0.1)": (0.416, 2.62, 0.091, 0.581),
            "bayes_privileged+cCRC(a=0.1)": (0.419, 3.20, 0.082, 0.441),
            "hindsight_oracle": (0.443, 1.54, 0.0, 0.861)}
    for name, (s, c, f, w) in rows.items():
        _close(_m(p, name, "success"), s, 5e-4, name + " success")
        _close(_m(p, name, "mean_calls"), c, 5e-3, name + " calls")
        _close(_m(p, name, "FAR_cond"), f, 5e-4, name + " FAR_cond")
        _close(_m(p, name, "waste_reduction"), w, 5e-4, name + " waste")
    for a, want in ((0.05, 0.045), (0.1, 0.091), (0.2, 0.189)):
        _close(_m(p, f"learned_hgb+cCRC(a={a})", "FAR_cond"), want, 5e-4, f"cCRC a={a}")
    _close(p["learned_hgb+cCRC(a=0.1)"]["FAR_cond"]["hi"], 0.15, 5e-3, "97.5th pct FAR_cond")
    d = _j("E1_main_E7_leakage.json")["predictor_diagnostics_on_test_pool"]
    _close(d["observable"]["auroc"], 0.870, 5e-4, "AUROC observable")
    _close(d["leak_recoverability_tag"]["auroc"], 0.997, 5e-4, "AUROC with leaked tag")
    _close(_m(p, "learned_hgb[leak_recoverability_tag]+cCRC(a=0.1)", "waste_reduction"), 0.85, 5e-3, "leak waste")


def test_sweeps_and_held_out():
    e3, e4, e2 = _j("E3_pi.json"), _j("E4_q.json"), _j("E2_budget.json")
    for pi, want in (("0.1", 0.17), ("0.5", 0.33), ("0.9", 0.99)):
        _close(_m(e3[pi]["policies"], "learned_hgb+CRC(a=0.1)", "FAR_cond"), want, 5e-3, f"joint CRC pi={pi}")
    for d in (e2, e3, e4):  # in-distribution: conditional CRC mean FAR_cond <= alpha on every slice
        for k, v in d.items():
            assert _m(v["policies"], "learned_hgb+cCRC(a=0.1)", "FAR_cond") <= 0.1, k
    _close(_m(e4["0.2"]["policies"], "retry_3", "FAR_cond"), 0.007, 5e-4, "retry3 q=0.2")
    _close(_m(e4["0.8"]["policies"], "retry_3", "FAR_cond"), 0.319, 5e-4, "retry3 q=0.8")
    _close(_m(e4["0.8"]["policies"], "learned_hgb+cCRC(a=0.1)", "waste_reduction"), 0.36, 5e-3, "q=0.8 waste")
    for prof, want in (("tight", 0.30), ("standard", 0.52), ("relaxed", 0.70)):
        _close(_m(e2[prof]["policies"], "learned_hgb+cCRC(a=0.1)", "waste_reduction"), want, 5e-3, prof)
    e5, e6 = _j("E5_heldout_family.json"), _j("E6_heldout_domain.json")
    _close(_m(e5["timeout"]["policies"], "learned_hgb+cCRC(a=0.1)", "FAR_cond"), 0.156, 5e-4, "E5 timeout")
    _close(_m(e6["retrieval"]["policies"], "learned_hgb+cCRC(a=0.1)", "FAR_cond"), 0.178, 5e-4, "E6 retrieval")
    b = _j("05_breakdown.json")
    _close(b["timeout/transient"]["learned_FARcond"], 0.21, 5e-3, "breakdown timeout/transient")
    _close(b["adversarial_error/transient"]["learned_FARcond"], 0.15, 5e-3, "breakdown adversarial/transient")


def test_E8_and_classification():
    s = json.loads((R / "q_shift" / "summary.json").read_text(encoding="utf-8"))
    _close(s["0.8"]["frozen_learned"]["FAR_cond"]["mean"], 0.324, 5e-4, "E8 frozen q=0.8")
    assert s["0.8"]["frozen_learned"]["frac_splits_FARcond_gt_alpha"] == 1.0
    _close(s["0.2"]["frozen_learned"]["FAR_cond"]["mean"], 0.028, 5e-4, "E8 frozen q=0.2")
    _close(s["0.8"]["recalibrated_learned"]["FAR_cond"]["mean"], 0.096, 5e-4, "E8 recalibrated q=0.8")
    c = _j("classification_metrics.json")
    gb, lr = c["models"]["gradient_boosting"], c["models"]["logistic_regression"]
    assert gb["T_0.5"]["all_decision_points"]["confusion"] == {"TN": 7340, "FP": 153, "FN": 1017, "TP": 150}
    assert lr["T_0.5"]["all_decision_points"]["confusion"] == {"TN": 7393, "FP": 100, "FN": 1036, "TP": 131}
    assert gb["T_train"]["all_decision_points"]["confusion"] == {"TN": 5683, "FP": 1810, "FN": 187, "TP": 980}
    assert lr["T_train"]["all_decision_points"]["confusion"] == {"TN": 5461, "FP": 2032, "FN": 135, "TP": 1032}
    _close(gb["T_0.5"]["first_decision_point_only"]["auroc"], 0.693, 5e-4, "first-dp AUROC GB")
    _close(lr["T_0.5"]["first_decision_point_only"]["auroc"], 0.703, 5e-4, "first-dp AUROC LR")
