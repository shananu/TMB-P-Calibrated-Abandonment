"""Integrity tests for scripts/08_classification_metrics.py.
Run: python -m tests test_classification   (or `python -m tests` for the whole suite)"""
import json, pickle
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from tmbp.experiments import FEATURE_SETS, Learned, dp_table

def test_no_leaky_features():
    cats, nums = FEATURE_SETS["observable"]
    assert not [c for c in cats + nums if c.startswith("leak")]

def test_refit_reproduces_saved_predictions_and_alignment():
    eps = pickle.load(open("data/tmbp_pi0.5_q0.5/episodes.pkl", "rb"))
    saved = json.load(open("results/E1_main_E7_leakage.json"))["predictor_diagnostics_on_test_pool"]
    rows, y = dp_table(eps["pool"])
    assert len(rows) == len(y) == sum(len(e.dps) for e in eps["pool"])
    for kind, key in (("hgb", "observable"), ("lr", "learned_lr")):
        m = Learned("observable", model=kind).fit(eps["train"])
        p = m.model.predict_proba(m.enc.transform(rows))[:, 1]
        assert abs(roc_auc_score(y, p) - saved[key]["auroc"]) < 1e-12
        assert abs(brier_score_loss(y, p) - saved[key]["brier"]) < 1e-12

def test_reported_confusion_is_consistent():
    r = json.load(open("results/classification_metrics.json"))
    n = r["support"]["test_decision_points"]
    for name, m in r["models"].items():
        for t in ("T_0.5", "T_train"):
            a = m[t]["all_decision_points"]; c = a["confusion"]
            assert c["TN"] + c["FP"] + c["FN"] + c["TP"] == n["n"]
            assert c["TP"] + c["FN"] == n["Y1"] and c["TN"] + c["FP"] == n["Y0"]
            assert np.isclose(a["accuracy"], (c["TP"] + c["TN"]) / n["n"])
            assert np.isclose(a["recall_sensitivity"], c["TP"] / (c["TP"] + c["FN"]))
        assert m["T_train"]["all_decision_points"]["threshold"] != 0.5  # selected on train, not default

if __name__ == "__main__":
    test_no_leaky_features(); test_refit_reproduces_saved_predictions_and_alignment(); test_reported_confusion_is_consistent()
    print("classification tests passed")
