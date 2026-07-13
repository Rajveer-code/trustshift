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


def fig5_remediation():
    rows = []
    for dom in DOMAINS:
        fp = P["out"] / f"remediation_{dom}.json"
        if not fp.exists():
            continue
        r = json.loads(fp.read_text())
        for tgt, m in r["targets"].items():
            rows.append((f"{dom}/{tgt.replace('target_', '')}", m["ece_L0"], m["ece_isotonic"],
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


def main():
    t2 = _t2()
    if t2.empty:
        print("no T2_master.csv — run synthesis.tables first"); return
    fig2_shift_bars(t2)
    fig3_headline(t2)
    fig5_remediation()
    print(f"wrote figures to {FDIR}: " + ", ".join(p.name for p in sorted(FDIR.glob('*.png'))))


if __name__ == "__main__":
    main()
