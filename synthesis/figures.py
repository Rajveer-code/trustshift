"""Paper figures (PLAN.md Task 9). 300 dpi PNG+PDF, serif, Okabe-Ito, no chartjunk.

F2 source-vs-target AUC & ECE bars (all domains)   F3 dAUC-vs-dG scatter + meta-regression (HEADLINE)
F5 remediation: ECE/AUC/gap before vs after isotonic   All read results/*.json — no hardcoded numbers.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P, OKABE  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titleweight": "bold",
                     "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "lines.linewidth": 2.2,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
FDIR = P["out"] / "figures"
FDIR.mkdir(exist_ok=True)
DOMAINS = ["clinical", "nlp", "lending", "security"]


def _save(fig, name):
    fig.savefig(FDIR / f"{name}.png")
    fig.savefig(FDIR / f"{name}.pdf")
    plt.close(fig)


def _t2() -> pd.DataFrame:
    fp = P["out"] / "tables" / "T2_master.csv"
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame()


def fig2_shift_bars(t2):
    """Slopegraph: each shift-point is a line from its source value to its target value,
    colored by domain. Reads the transfer directly (down = degrades, flat = robust)."""
    doms = sorted(t2.domain.unique())
    cmap = {d: OKABE[i % len(OKABE)] for i, d in enumerate(doms)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    panels = [("auc_src", "auc_tgt", "Discrimination (AUC)", "AUC", False),
              ("ece_src", "ece_tgt", "Calibration (ECE, lower is better)", "ECE", True)]
    for ax, (cs, ct, title, ylab, _) in zip(axes, panels):
        for _, r in t2.iterrows():
            ax.plot([0, 1], [r[cs], r[ct]], "-o", color=cmap[r.domain], lw=1.8,
                    ms=6, alpha=0.85, mec="white", mew=0.8)
        ax.set_xlim(-0.35, 1.35); ax.set_xticks([0, 1])
        ax.set_xticklabels(["source\n(in-domain)", "target\n(deployed)"], fontsize=11)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.15); ax.margins(y=0.08)
    handles = [plt.Line2D([0], [0], color=cmap[d], lw=3, marker="o", label=d) for d in doms]
    axes[0].legend(handles=handles, loc="lower left", framealpha=0.9)
    fig.suptitle("Does trustworthiness transfer? Each line is one deployment shift",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    _save(fig, "fig2_shift_bars")


def fig3_headline(t2):
    """dAUC vs dG scatter, colored by domain, with the M3 meta-regression line."""
    d = t2.dropna(subset=["delta_gap_auc"]).copy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for i, dom in enumerate(sorted(d.domain.unique())):
        s = d[d.domain == dom]
        ax.scatter(s["delta_auc"], s["delta_gap_auc"], s=150, color=OKABE[i % len(OKABE)],
                   label=dom, edgecolor="k", linewidth=1.0, zorder=3)
    meta_fp = P["out"] / "meta_analysis.json"
    if meta_fp.exists():
        m = json.loads(meta_fp.read_text()).get("M3_gap_vs_accuracyloss", {})
        if m.get("slope") is not None:
            xs = np.linspace(d["delta_auc"].min(), d["delta_auc"].max(), 50)
            b1 = m["slope"]; b0 = np.mean(d["delta_gap_auc"]) - b1 * np.mean(d["delta_auc"])
            ax.plot(xs, b1 * xs + b0, "--", color="0.25", lw=2.0,
                    label=f"cross-domain OLS: slope {b1:.2f}, $R^2$={m.get('r2', 0):.2f}", zorder=2)
    ax.axhline(0, color="grey", lw=1.0, zorder=1); ax.axvline(0, color="grey", lw=1.0, zorder=1)
    ax.set_xlabel("$\\Delta$ aggregate AUC  (source $-$ target)")
    ax.set_ylabel("$\\Delta$ subgroup-gap AUC  (target $-$ source)")
    ax.set_title("Subgroup loss vs. accuracy loss across shift-points")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.15)
    _save(fig, "fig3_headline_scatter")


def _remediation_files(dom):
    """Single-experiment domains write remediation_{dom}.json; multi-experiment domains
    (lending: temporal + geographic are different experiments) write one
    remediation_{dom}_{model}.json per experiment (see audit/primary_model.py)."""
    single = P["out"] / f"remediation_{dom}.json"
    if single.exists():
        return [json.loads(single.read_text())]
    return [json.loads(fp.read_text()) for fp in sorted(P["out"].glob(f"remediation_{dom}_*.json"))]


def fig5_remediation():
    rows = []
    for dom in DOMAINS:
        for r in _remediation_files(dom):
            model_tag = f"[{r['model']}]" if len(_remediation_files(dom)) > 1 else ""
            for tgt, m in r["targets"].items():
                rows.append((f"{dom}{model_tag}/{tgt.replace('target_', '')}", m["ece_L0"], m["ece_isotonic"],
                             m["auc_L0"], m["auc_isotonic"]))
    if not rows:
        return
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, max(3, 0.5 * len(rows))))
    a1.barh(y - 0.2, [r[1] for r in rows], 0.4, label="no fix", color=OKABE[3])
    a1.barh(y + 0.2, [r[2] for r in rows], 0.4, label="isotonic", color=OKABE[2])
    a1.set_yticks(y); a1.set_yticklabels(labels, fontsize=6); a1.set_title("ECE (recalibration fixes this)")
    a1.legend(fontsize=8); a1.grid(axis="x", alpha=0.3)
    a2.barh(y - 0.2, [r[3] for r in rows], 0.4, label="no fix", color=OKABE[3])
    a2.barh(y + 0.2, [r[4] for r in rows], 0.4, label="isotonic", color=OKABE[2])
    a2.set_yticks(y); a2.set_yticklabels([]); a2.set_title("AUC (recalibration does NOT change this)")
    a2.legend(fontsize=8); a2.grid(axis="x", alpha=0.3)
    fig.suptitle("Remediation ladder L1: recalibration restores calibration, not discrimination",
                 fontweight="bold")
    _save(fig, "fig5_remediation")


def fig1_taxonomy():
    """Conceptual audit-flow diagram (Phase 4 redesign). Evidence-first: deployment -> label-free
    screening -> available evidence -> mechanism hypothesis / inconclusive (parallel, not a failure
    branch) -> multi-axis trustworthiness audit -> remediation or escalation. The four trustworthiness
    axes are shown as a separate, jointly-audited panel with one unlabeled arrow from the audit step --
    never as four arrows from individual mechanism hypotheses, so the figure does not assert a
    deterministic mechanism-to-axis mapping the manuscript's own evidence does not support."""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(9, 8.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.1, 10.4)
    ax.axis("off")
    fig.subplots_adjust(top=0.94, bottom=0.03, left=0.02, right=0.98)

    def box(cx, cy, w, h, text, fc, ec=OKABE[0], dashed=False, fontsize=10.5, fontweight="bold"):
        p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                            boxstyle="round,pad=0.08,rounding_size=0.12",
                            linewidth=1.6, edgecolor=ec, facecolor=fc,
                            linestyle="dashed" if dashed else "solid")
        ax.add_patch(p)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                 fontweight=fontweight, color="#1a1a1a", wrap=True)
        return (cx, cy, w, h)

    def arrow(b_from, b_to, style="-|>", color="#444444", lw=1.4, connectionstyle=None):
        x0, y0, w0, h0 = b_from
        x1, y1, w1, h1 = b_to
        start = (x0, y0 - h0 / 2) if y0 > y1 else (x0, y0 + h0 / 2)
        end = (x1, y1 + h1 / 2) if y0 > y1 else (x1, y1 - h1 / 2)
        a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14,
                             color=color, linewidth=lw, connectionstyle=connectionstyle,
                             shrinkA=2, shrinkB=2)
        ax.add_patch(a)

    light = "#eef4fb"
    b_deploy = box(3.0, 9.9, 3.6, 0.7, "Deployment", light, ec="#333333", fontweight="bold")
    b_screen = box(3.0, 8.55, 4.4, 0.9,
                    "Stage A -- unlabeled screening\n" + r"$\Delta\pi$, AUC$_{dc}$",
                    "#dcecfb", ec=OKABE[0])
    b_evid = box(3.0, 7.2, 4.4, 0.9,
                  "Available evidence\n(target labels + overlap?\n"
                  r"$\to$ confirmatory reweighting probe)",
                  "#eef4fb", ec=OKABE[0], fontsize=9.5, fontweight="normal")
    arrow(b_deploy, b_screen)
    arrow(b_screen, b_evid)

    b_hyp = box(1.55, 5.7, 2.8, 0.9, "Mechanism\nhypothesis\n(label / covariate /\nconcept / mixed)",
                "#fde9d0", ec=OKABE[1], fontsize=9)
    b_inc = box(4.55, 5.7, 2.8, 0.9, "Inconclusive\n(first-class outcome,\nnot a failure)",
                "#f5f5f5", ec="#777777", dashed=True, fontsize=9.5)
    arrow(b_evid, b_hyp, connectionstyle="arc3,rad=-0.15")
    arrow(b_evid, b_inc, connectionstyle="arc3,rad=0.15")

    b_audit = box(3.0, 4.1, 5.6, 0.9, "Multi-axis trustworthiness audit", "#e3f2e9", ec=OKABE[2])
    arrow(b_hyp, b_audit, connectionstyle="arc3,rad=0.15")
    arrow(b_inc, b_audit, connectionstyle="arc3,rad=-0.15")

    b_remed = box(3.0, 2.5, 5.6, 0.9, "Remediation / escalation", "#f7e3ee", ec=OKABE[4])
    arrow(b_audit, b_remed)

    axis_labels = ["Discrimination", "Operating point", "Calibration", "Subgroup\nreliability"]
    axis_boxes = []
    for i, lab in enumerate(axis_labels):
        bx = box(7.9, 5.55 - i * 1.05, 3.0, 0.75, lab, "#f2f2f2", ec="#666666",
                  fontsize=9.5, fontweight="normal")
        axis_boxes.append(bx)
    bracket = mpatches.FancyBboxPatch((6.25, 1.55), 0.12, 4.6, boxstyle="round,pad=0.0",
                                       linewidth=1.2, edgecolor="#666666", facecolor="#666666")
    ax.add_patch(bracket)
    ax.annotate("", xy=(6.37, 3.85), xytext=(5.8, 4.1),
                arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.4,
                                 connectionstyle="arc3,rad=-0.2"))
    ax.text(7.9, 6.55, "audited jointly, not\nfrom separate mechanism arrows", ha="center",
             va="bottom", fontsize=8.3, style="italic", color="#555555")

    ax.text(0.15, 0.35,
            "Dashed border = evidence does not support a firm mechanism label.\n"
            "No arrow in this figure implies a deterministic mechanism "
            r"$\to$ failure-axis mapping.",
            fontsize=8.3, style="italic", color="#555555", va="bottom")

    fig.suptitle("The TrustShift audit flow", fontweight="bold", fontsize=15, y=0.995)
    _save(fig, "fig1_taxonomy")


def main():
    fig1_taxonomy()
    t2 = _t2()
    if t2.empty:
        print("no T2_master.csv — run synthesis.tables first"); return
    fig2_shift_bars(t2)
    fig3_headline(t2)
    fig5_remediation()
    print(f"wrote figures to {FDIR}: " + ", ".join(p.name for p in sorted(FDIR.glob('*.png'))))


if __name__ == "__main__":
    main()
