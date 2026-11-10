"""Paper figures (static PNG). Palette: validated 4-slot categorical; lines are direct-labeled + distinct markers."""
import json
import _bootstrap  # noqa: F401  (paths, cwd)
from _bootstrap import RESULTS, guard
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = RESULTS; F = R / "figures"
guard(*(F / f for f in ("fig1_frontier.png", "fig2_q_drift.png", "fig3_joint_vs_conditional.png")))
C = {"learned_hgb": "#2a78d6", "table_tool_code_attempt_budget": "#eb6834", "bayes_privileged": "#1baf7a", "beta_tool_code": "#eda100"}
LBL = {"learned_hgb": "Learned (HGB)", "table_tool_code_attempt_budget": "Lookup table (tool, code, attempt, budget)",
       "bayes_privileged": "Bayes (privileged π, q)", "beta_tool_code": "Beta per (tool, code)"}
MK = {"learned_hgb": "o", "table_tool_code_attempt_budget": "s", "bayes_privileged": "^", "beta_tool_code": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
e1 = json.loads((R / "E1_main_E7_leakage.json").read_text())

# Fig 1: risk-cost frontier
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for k in C:
    pts = sorted(e1["frontiers"][k], key=lambda p: p["FAR_cond"])
    best, xs, ys = -1, [], []
    for p in pts:  # upper envelope
        if p["waste_reduction"] > best: best = p["waste_reduction"]; xs.append(p["FAR_cond"]); ys.append(best)
    ax.step(xs, ys, where="post", color=C[k], lw=2, label=LBL[k])
for k, r in e1["rule_points"].items():
    if k in ("never_stop", "hindsight_oracle"): continue
    ax.plot(r["FAR_cond"], r["waste_reduction"], "x", color=INK, ms=8, mew=2)
    ax.annotate(k.replace("_", " "), (r["FAR_cond"], r["waste_reduction"]), xytext=(5, -12), textcoords="offset points", color=MUTED, fontsize=8)
ax.axhline(e1["rule_points"]["hindsight_oracle"]["waste_reduction"], color=MUTED, ls=":", lw=1)
ax.text(0.62, 0.875, "hindsight oracle", color=MUTED, fontsize=8)
ax.axvline(0.1, color=MUTED, ls="--", lw=1); ax.text(0.105, 0.03, "α = 0.10", color=MUTED, fontsize=8)
ax.set(xlim=(0, 1.02), ylim=(0, 0.95), xlabel="False-abandonment rate among would-succeed episodes (FAR_cond)",
       ylabel="Reduction in wasted tool calls vs never-stop")
ax.grid(True, color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Stop-policy frontiers on TMB-P (π = 0.5, q = 0.5); × = fixed rules", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(F / "fig1_frontier.png", dpi=200); plt.close(fig)

# Fig 2: fixed rules drift with q; calibrated policy holds alpha
e4 = json.loads((R / "E4_q.json").read_text()); qs = sorted(e4, key=float)
series = {"retry_3": ("Retry-3 (fixed)", "#eb6834", "s"), "error_type_rule": ("Error-type rule (fixed)", "#eda100", "D"),
          "learned_hgb+cCRC(a=0.1)": ("Learned + conditional CRC, α = 0.1", "#2a78d6", "o")}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for k, (lab, col, m) in series.items():
    y = [e4[q]["policies"][k]["FAR_cond"]["mean"] for q in qs]
    ax.plot([float(q) for q in qs], y, color=col, marker=m, ms=8, lw=2, label=lab)
    ax.annotate(lab, (float(qs[-1]), y[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
ax.axhline(0.1, color=MUTED, ls="--", lw=1)
ax.set(xlabel="Per-call failure probability of transient faults (q)", ylabel="FAR_cond (mean over 50 splits)", xlim=(0.15, 1.25), ylim=(0, 0.6))
ax.set_xticks([0.2, 0.5, 0.8]); ax.grid(True, color=GRID, lw=0.8)
ax.set_title("Fixed retry rules drift with fault stickiness; calibration holds α", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(F / "fig2_q_drift.png", dpi=200); plt.close(fig)

# Fig 3: joint vs conditional CRC under varying pi
e3 = json.loads((R / "E3_pi.json").read_text()); pis = sorted(e3, key=float)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for k, lab, col, m in [("learned_hgb+CRC(a=0.1)", "Joint CRC (audit formulation)", "#eb6834", "s"),
                       ("learned_hgb+cCRC(a=0.1)", "Conditional CRC (corrected)", "#2a78d6", "o")]:
    y = [e3[p]["policies"][k]["FAR_cond"]["mean"] for p in pis]
    ax.plot([float(p) for p in pis], y, color=col, marker=m, ms=8, lw=2, label=lab)
ax.axhline(0.1, color=MUTED, ls="--", lw=1); ax.text(0.1, 0.12, "α = 0.10", color=MUTED, fontsize=8)
ax.set(xlabel="Proportion of persistent faults (π)", ylabel="FAR_cond (mean over 50 splits)", ylim=(0, 1.05))
ax.grid(True, color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Joint risk lets abandonment of would-succeed tasks grow with π", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(F / "fig3_joint_vs_conditional.png", dpi=200); plt.close(fig)
print("figures written")
