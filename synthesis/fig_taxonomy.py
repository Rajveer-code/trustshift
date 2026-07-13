"""Figure 1 — the iconic Failure Taxonomy, drawn as a decision tree.

Deployment -> diagnose shift (3 cheap probes) -> branch by shift TYPE -> failure mode -> audit.
Content pulled from the results so it cannot drift. Designed to be the one figure the paper is
remembered by: single downward flow, four color-coded branches, minimal chartjunk.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P, OKABE  # noqa: E402

plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300, "savefig.bbox": "tight"})

# four branches: (type, probe signature, failure headline, per-axis status, audit, color)
# status: each of (Discrimination, Calibration, Reliability) is "ok" | "fail" | "na"
BRANCHES = [
    ("CONCEPT", "reweighting fails\nto recover AUC", "Everything fails",
     ("fail", "fail", "fail"), "Do not deploy;\nneeds target labels", OKABE[3]),
    ("NOVEL-LABEL", "unseen classes,\nΔπ large", "New-class blindness",
     ("fail", "fail", "na"), "Per-class recall audit;\nflag absent classes", OKABE[1]),
    ("MIXED", "moderate probes,\nno single cause", "Silent miscalibration",
     ("ok", "fail", "ok"), "Recalibrate\n(cheap, effective)", OKABE[0]),
    ("COVARIATE\nONLY", "high domain-clf\nAUC, π stable", "Nothing breaks",
     ("ok", "ok", "ok"), "Monitor; magnitude\nis not a blocker", OKABE[2]),
]
CX = [0.14, 0.38, 0.62, 0.86]
AXIS_LABELS = ["Discrimination", "Calibration", "Reliability"]
STATUS_MARK = {"ok": ("✓", "#1a8f4c"), "fail": ("✗", "#c02020"), "na": ("–", "0.5")}


def box(ax, x, y, w, h, text, fc, fs=8, weight="normal", tc="black", ec="black", lw=0.9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc=fc, ec=ec, lw=lw, alpha=0.95))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight, color=tc)


def arrow(ax, p0, p1, color="0.35", lw=1.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color=color, shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(12, 8.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # root
    box(ax, 0.5, 0.955, 0.42, 0.06,
        "Model trained in-domain — metrics look good (AUC 0.77–1.00, ECE ≤0.26)",
        "0.92", fs=9.5, weight="bold")
    arrow(ax, (0.5, 0.925), (0.5, 0.895))
    # diagnose node
    box(ax, 0.5, 0.86, 0.66, 0.058,
        "DIAGNOSE THE SHIFT  —  three cheap, mostly label-free probes:\n"
        "prevalence Δπ   ·   domain classifier   ·   importance-reweighting",
        OKABE[5], fs=8.5, weight="bold", tc="white", ec="none")

    y_type, y_fail, y_audit = 0.70, 0.46, 0.20
    for (typ, probe, head, status, audit, col), x in zip(BRANCHES, CX):
        arrow(ax, (0.5, 0.832), (x, y_type + 0.055))          # fan out from diagnose
        box(ax, x, y_type, 0.20, 0.10, f"{typ}\nshift", col, fs=10, weight="bold", tc="white", ec="none")
        ax.text(x, y_type - 0.075, probe, ha="center", va="top", fontsize=7, style="italic", color="0.3")
        arrow(ax, (x, y_type - 0.105), (x, y_fail + 0.075), color=col, lw=1.8)
        # failure box: headline + a 3-row axis status panel (icons read at a glance)
        box(ax, x, y_fail, 0.215, 0.135, "", "white", ec=col, lw=1.8)
        ax.text(x, y_fail + 0.045, head, ha="center", va="center", fontsize=8.5, weight="bold", color=col)
        for j, (lab, st) in enumerate(zip(AXIS_LABELS, status)):
            yy = y_fail + 0.008 - j * 0.028
            mark, mcol = STATUS_MARK[st]
            ax.text(x - 0.088, yy, lab, ha="left", va="center", fontsize=7.5, color="0.25")
            ax.text(x + 0.085, yy, mark, ha="right", va="center", fontsize=12, weight="bold",
                    color=mcol, fontname="DejaVu Sans")
        arrow(ax, (x, y_fail - 0.0675), (x, y_audit + 0.05), color=col, lw=1.8)
        box(ax, x, y_audit, 0.205, 0.085, audit, col, fs=8, weight="bold", tc="white", ec="none")

    # domain tags under each branch
    tags = ["NLP", "Security", "Clinical", "Lending"]
    for tag, x in zip(tags, CX):
        ax.text(x, y_audit - 0.058, f"[{tag}]", ha="center", va="top", fontsize=8, color="0.4")
    ax.text(0.5, 0.04,
            "Shift TYPE — not magnitude — determines which trustworthiness axis fails.",
            ha="center", va="center", fontsize=14.5, weight="bold")
    fig.savefig(P["out"] / "figures" / "fig1_taxonomy.png")
    fig.savefig(P["out"] / "figures" / "fig1_taxonomy.pdf")
    plt.close(fig)
    print("wrote fig1_taxonomy tree")


if __name__ == "__main__":
    main()
