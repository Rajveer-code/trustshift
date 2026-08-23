"""Phase 3A/3B analysis for T3.1 (docs/PHASE3_DESIGN.md ## T3.1 + 'Primary and secondary
endpoints' + H1-H3 falsification criteria). Reads lending_shiftpoints.csv (written by
domains/lending/phase3_shiftpoints.py) and produces:
  - results/lending_phase3_analysis.json   (primary fingerprint + secondary slope, all 4 axes)
  - results/figures/fig_phase3_lending_severity_surface.{png,pdf}
  - results/phase3_T31_verification.json   (Phase 3A data-generation verification)
  - results/phase3_T31_hypothesis_status.json  (Phase 3B: H1/H2 status against frozen criteria)

PRE-SPECIFIED analysis protocol (written before this script is ever run against real T3.1
output -- classification thresholds below are fixed now, not tuned after seeing results):

Primary endpoint (per axis in {delta_auc, delta_macro_f1, delta_ece, delta_gap_auc} -- the same
four canonical failure types as manuscript Table `tab:fingerprint`): the relationship between
CONTINUOUS auc_dc and that axis's delta, characterized by (a) Spearman rho with a cluster
(state) bootstrap CI, (b) a fine quantile-binned mean curve (secondary VISUALIZATION of the
continuous relationship only, never the statistical basis -- PHASE3_DESIGN.md), (c) a three-way
classification:
    "flat"          if the cluster-bootstrap 95% CI on Spearman rho contains 0.
    "monotonic"     if the CI excludes 0 AND >=80% of consecutive fine-bin-mean steps move in
                    the same direction as the overall rho sign (tolerance = 0.1 * std of the
                    bin means, to avoid penalizing sampling noise between adjacent bins).
    "non-monotonic" if the CI excludes 0 but the 80% consistency bar is not met (a real
                    association exists but the shape is not simply monotonic).
Fine-bin count = clip(n_cells // 15, 4, 10) quantile bins on auc_dc -- fixed formula, not chosen
per axis after looking at the data.

Secondary endpoint: within-domain OLS slope + R^2 of delta_X on continuous auc_dc (point
estimate via numpy.polyfit; uncertainty via the SAME cluster bootstrap, never a closed-form/IID
OLS CI -- PHASE3_DESIGN.md explicitly rejects that).

H2 combining rule (fixed now, applied mechanically in _h2_status below):
    Falsified if >=3 of 4 axes have R^2 > 0.7 AND a slope bootstrap CI excluding 0 (magnitude
        alone explains the failure pattern well and consistently).
    Supported if >=3 of 4 axes have R^2 < 0.3 OR are classified "flat" (within-mechanism
        variance dominates any magnitude-driven trend).
    Mixed otherwise.
R^2 thresholds (0.3 / 0.7) are the conventional weak/strong bands for a single-predictor fit,
fixed before this script ever saw T3.1 output.

H1 (mechanism produces distinguishable fingerprints) is NOT assessable from T3.1 alone -- T3.1
manipulates only ONE mechanism (covariate-like shift within lending: geography/time). H1
requires comparing across >=2 mechanisms and is deferred to the Phase 3H global synthesis, once
T3.2 (novel-mechanism) and T3.3 (prior/concept) results exist. Recording anything else here would
be a premature, unearned verdict.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P, N_BOOT, SEEDS_NEW  # noqa: E402
from synthesis.figures import _save, OKABE  # noqa: E402 (reuses house style: 300dpi, serif, Okabe-Ito)
import matplotlib.pyplot as plt  # noqa: E402

AXES = ["delta_auc", "delta_macro_f1", "delta_f1_positive", "delta_ece", "delta_brier", "delta_gap_auc"]
AXIS_LABEL = {"delta_auc": "$\\Delta$AUC (discrimination)",
              "delta_macro_f1": "$\\Delta$macro-F1 (operating point, average=\"macro\")",
              "delta_f1_positive": "$\\Delta$F1$_{positive}$ (approved-class-only F1, diagnostic)",
              "delta_ece": "$\\Delta$ECE (calibration)",
              "delta_brier": "$\\Delta$Brier (calibration, diagnostic)",
              "delta_gap_auc": "$\\Delta$G (subgroup gap, race_black)"}
# 2026-08-23 (round-4 hostile audit, item H): report ALL SIX axes' fingerprint individually and
# transparently (never collapse into one generic "all axes flat" line). The H2 SUPPORTED/MIXED/
# FALSIFIED verdict, however, votes over only the four NON-REDUNDANT canonical trustworthiness
# dimensions this benchmark uses everywhere else (Table `tab:fingerprint`): discrimination (AUC),
# calibration (ECE), operating-point (macro_f1 -- the CORRECTED metric), subgroup (gap_auc).
# delta_f1_positive and delta_brier are reported as diagnostic/supplementary axes only; including
# them in the same vote would double-count operating-point (f1_positive and macro_f1 measure the
# same underlying threshold-based classification) and calibration (Brier and ECE are both
# calibration measures) rather than adding an independent dimension of evidence.
H2_VOTING_AXES = ["delta_auc", "delta_macro_f1", "delta_ece", "delta_gap_auc"]
R2_WEAK, R2_STRONG = 0.3, 0.7
RHO_BOOT_SEED = 42


def _n_bins(n_cells: int) -> int:
    return int(np.clip(n_cells // 15, 4, 10))


def _fine_bins(auc_dc: np.ndarray, delta: np.ndarray, n_bins: int) -> list[dict]:
    edges = np.quantile(auc_dc, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.digitize(auc_dc, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"n": int(m.sum()), "auc_dc_mean": float(auc_dc[m].mean()),
                    "auc_dc_range": [float(auc_dc[m].min()), float(auc_dc[m].max())],
                    "delta_mean": float(delta[m].mean()), "delta_std": float(delta[m].std())})
    return out


def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or np.std(x) == 0:
        return dict(slope=float("nan"), intercept=float("nan"), r2=float("nan"))
    b1, b0 = np.polyfit(x, y, 1)
    yhat = b1 * x + b0
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(slope=float(b1), intercept=float(b0),
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))


def _classify(auc_dc, delta, rho, rho_lo, rho_hi, n_bins) -> tuple[str, float | None]:
    if rho_lo <= 0 <= rho_hi:
        return "flat", None
    edges = np.quantile(auc_dc, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.digitize(auc_dc, edges[1:-1]), 0, n_bins - 1)
    means = np.array([delta[idx == b].mean() for b in range(n_bins) if (idx == b).sum() > 0])
    if len(means) < 2:
        return "non-monotonic", 0.0
    diffs = np.diff(means)
    expected_sign = 1 if rho > 0 else -1
    tol = 0.1 * float(np.std(means))
    frac_consistent = float(np.mean(diffs * expected_sign >= -tol))
    return ("monotonic" if frac_consistent >= 0.8 else "non-monotonic"), frac_consistent


def cluster_bootstrap(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = RHO_BOOT_SEED) -> dict:
    """State cluster bootstrap (PHASE3_DESIGN.md T3.1, exact procedure): resample INCLUDED
    states with replacement, keep every (year x seed) row for each resampled state, recompute
    per-axis Spearman rho / OLS slope / fraction-degrading on the resampled panel, repeat
    n_boot times, report 2.5th/97.5th percentiles."""
    groups = {s: sub.reset_index(drop=True) for s, sub in df.groupby("state_fips")}
    state_list = np.array(list(groups.keys()))
    rng = np.random.default_rng(seed)
    boot = {a: {"rho": [], "slope": [], "frac_degrading": []} for a in AXES}
    for _ in range(n_boot):
        resampled_states = rng.choice(state_list, size=len(state_list), replace=True)
        b = pd.concat([groups[s] for s in resampled_states], ignore_index=True)
        x = b["auc_dc"].to_numpy()
        for a in AXES:
            y = b[a].to_numpy()
            if np.std(x) == 0:
                continue
            boot[a]["rho"].append(spearmanr(x, y).correlation)
            boot[a]["slope"].append(_ols(x, y)["slope"])
            boot[a]["frac_degrading"].append(float(np.mean(y > 0)))
    return boot


def fingerprint(df: pd.DataFrame, boot: dict) -> dict:
    x = df["auc_dc"].to_numpy()
    n_bins = _n_bins(len(df))
    out = {"n_cells": int(len(df)), "n_bins": n_bins, "axes": {}}
    for a in AXES:
        y = df[a].to_numpy()
        rho = float(spearmanr(x, y).correlation)
        rho_lo, rho_hi = (float(v) for v in np.percentile([r for r in boot[a]["rho"] if r == r], [2.5, 97.5]))
        slope_lo, slope_hi = (float(v) for v in np.percentile([s for s in boot[a]["slope"] if s == s], [2.5, 97.5]))
        frac_lo, frac_hi = (float(v) for v in np.percentile(boot[a]["frac_degrading"], [2.5, 97.5]))
        cls, consistency = _classify(x, y, rho, rho_lo, rho_hi, n_bins)
        ols = _ols(x, y)
        out["axes"][a] = {
            "label": AXIS_LABEL[a],
            "spearman_rho": rho, "spearman_rho_ci95": [rho_lo, rho_hi],
            "classification": cls, "bin_monotonicity_consistency": consistency,
            "ols_slope": ols["slope"], "ols_slope_ci95": [slope_lo, slope_hi], "r2": ols["r2"],
            "fraction_degrading": float(np.mean(y > 0)), "fraction_degrading_ci95": [frac_lo, frac_hi],
            "fine_bins": _fine_bins(x, y, n_bins),
        }
    return out


def _h2_status(fp: dict) -> dict:
    """Votes over H2_VOTING_AXES only (the 4 non-redundant canonical dimensions) -- see the
    module-level comment above H2_VOTING_AXES for why delta_f1_positive/delta_brier are reported
    but excluded from this specific vote."""
    axes = fp["axes"]
    n = len(H2_VOTING_AXES)
    maj = 3 if n == 4 else int(np.ceil(0.75 * n))
    strong = sum(1 for a in H2_VOTING_AXES if axes[a]["r2"] == axes[a]["r2"] and axes[a]["r2"] > R2_STRONG
                 and not (axes[a]["ols_slope_ci95"][0] <= 0 <= axes[a]["ols_slope_ci95"][1]))
    weak = sum(1 for a in H2_VOTING_AXES if axes[a]["classification"] == "flat"
               or (axes[a]["r2"] == axes[a]["r2"] and axes[a]["r2"] < R2_WEAK))
    if strong >= maj:
        status = "falsified"
        reason = f"{strong}/{n} axes: R2>{R2_STRONG} with a slope CI excluding 0 -- magnitude alone explains the pattern well and consistently"
    elif weak >= maj:
        status = "supported"
        reason = f"{weak}/{n} axes: flat or R2<{R2_WEAK} -- within-mechanism variance dominates any magnitude-driven trend"
    else:
        status = "mixed"
        reason = f"{strong}/{n} axes strong, {weak}/{n} axes weak -- magnitude explains some but not all of the variance"
    return {"status": status, "reason": reason, "strong_axes": strong, "weak_axes": weak,
           "voting_axes": H2_VOTING_AXES, "majority_threshold": maj}


def figure(df: pd.DataFrame, fp: dict):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, a in zip(axes.flat, AXES):
        x = df["auc_dc"].to_numpy(); y = df[a].to_numpy()
        ax.scatter(x, y, s=14, color=OKABE[0], alpha=0.35, zorder=2, linewidth=0)
        bins = fp["axes"][a]["fine_bins"]
        bx = [b["auc_dc_mean"] for b in bins]; by = [b["delta_mean"] for b in bins]
        ax.plot(bx, by, "-o", color=OKABE[3], lw=2.2, ms=6, zorder=3, label="fine-bin mean")
        ax.axhline(0, color="grey", lw=1.0, zorder=1)
        cls = fp["axes"][a]["classification"]; rho = fp["axes"][a]["spearman_rho"]
        ax.set_title(f"{AXIS_LABEL[a]} -- {cls} ($\\rho$={rho:.2f})", fontsize=12)
        ax.set_xlabel("AUC$_{dc}$ (covariate-shift magnitude)"); ax.set_ylabel("$\\delta$ (source $-$ target, + = worse)")
        ax.grid(alpha=0.15)
    axes.flat[0].legend(loc="best", fontsize=9)
    fig.suptitle("T3.1 lending: failure fingerprint vs. continuous covariate-shift magnitude",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    _save(fig, "fig_phase3_lending_severity_surface")


def main():
    df = pd.read_csv(P["out"] / "lending_shiftpoints.csv")
    excl = json.loads((P["out"] / "lending_excluded_cells.json").read_text())

    boot = cluster_bootstrap(df)
    fp = fingerprint(df, boot)
    analysis = {
        "n_cells": len(df), "n_states_included": df["state_fips"].nunique(),
        "n_years": df["year"].nunique(), "n_seeds": df["seed"].nunique(),
        "bootstrap": {"method": "cluster (block) bootstrap over states, all (year,seed) rows "
                                "per resampled state kept together", "n_boot": N_BOOT},
        "fingerprint": fp,
    }
    (P["out"] / "lending_phase3_analysis.json").write_text(json.dumps(analysis, indent=2))
    figure(df, fp)

    verification = {
        "n_states": excl["n_states"], "n_years": excl["n_years"], "n_seeds": len(SEEDS_NEW),
        "n_cells": len(df) // len(SEEDS_NEW) if len(SEEDS_NEW) else 0,
        "n_cells_scored_rows": len(df), "n_excluded": excl["n_excluded"],
        "exclusion_reasons": excl["excluded"],
        "bootstrap_method": "cluster (block) bootstrap over included states with replacement; "
                            "every (year x seed) observation for a resampled state is kept "
                            "together; 2000 resamples; 2.5th/97.5th percentile CI "
                            "(docs/PHASE3_DESIGN.md T3.1 'Dependence / uncertainty method')",
        "primary_endpoint": "failure-fingerprint characterization against CONTINUOUS auc_dc "
                            "(fine quantile-binned mean curve as visualization only; "
                            "tercile/quantile binning is never the statistical basis) for each "
                            "of delta_auc, delta_macro_f1, delta_ece, delta_gap_auc",
        "secondary_endpoint": "within-domain OLS slope + R2 of each axis's delta on continuous "
                              "auc_dc, point estimate via OLS, uncertainty via the cluster "
                              "bootstrap above (never a closed-form/IID OLS CI)",
    }
    (P["out"] / "phase3_T31_verification.json").write_text(json.dumps(verification, indent=2))

    h2 = _h2_status(fp)
    hyp_status = {
        "H1_status": "not_assessable_from_T31_alone",
        "H1_note": ("T3.1 manipulates a single mechanism (covariate-like shift within lending: "
                    "state x year). H1 requires >=2 distinguishable mechanisms and is decided "
                    "at the Phase 3H global synthesis once T3.2/T3.3 exist."),
        "H2_status": h2["status"], "H2_reason": h2["reason"],
        "evidence": {a: {"classification": fp["axes"][a]["classification"],
                         "r2": fp["axes"][a]["r2"], "spearman_rho": fp["axes"][a]["spearman_rho"]}
                    for a in AXES},
        "effect_sizes": {a: {"ols_slope": fp["axes"][a]["ols_slope"],
                             "fraction_degrading": fp["axes"][a]["fraction_degrading"]} for a in AXES},
        "uncertainty": {a: {"spearman_rho_ci95": fp["axes"][a]["spearman_rho_ci95"],
                            "ols_slope_ci95": fp["axes"][a]["ols_slope_ci95"],
                            "fraction_degrading_ci95": fp["axes"][a]["fraction_degrading_ci95"]}
                        for a in AXES},
        "caveats": [
            "H2 combining rule (R2<0.3 weak / >0.7 strong, >=3-of-4-axes majority) votes over the "
            "4 non-redundant canonical dimensions (H2_VOTING_AXES: delta_auc, delta_macro_f1, "
            "delta_ece, delta_gap_auc), fixed before this script was run against real T3.1 "
            "output. delta_f1_positive and delta_brier are reported in full (see 'evidence' "
            "below) but excluded from the vote to avoid double-counting operating-point and "
            "calibration evidence (2026-08-23 F1-definition correction, round-4 hostile audit).",
            "T3.1 covers ONE mechanism only (covariate-like geographic/temporal shift); H2's "
            "verdict here applies to that mechanism, not to prior/concept/novel-mechanism shift.",
            f"n_cells={len(df)} rows across {excl['n_included']} included (state,year) cells x "
            f"{len(SEEDS_NEW)} seeds; {excl['n_excluded']} cells excluded (see "
            "lending_excluded_cells.json for exact reasons).",
        ],
    }
    (P["out"] / "phase3_T31_hypothesis_status.json").write_text(json.dumps(hyp_status, indent=2))

    print(f"T3.1 analysis: {len(df)} rows, {excl['n_included']} cells, {excl['n_excluded']} excluded")
    for a in AXES:
        ax = fp["axes"][a]
        print(f"  {a:16s} rho={ax['spearman_rho']:+.3f} CI={ax['spearman_rho_ci95']} "
              f"r2={ax['r2']:.3f} class={ax['classification']} frac_degrading={ax['fraction_degrading']:.3f}")
    print(f"H2 status: {h2['status']} -- {h2['reason']}")
    print("wrote lending_phase3_analysis.json, phase3_T31_verification.json, "
          "phase3_T31_hypothesis_status.json, fig_phase3_lending_severity_surface.{png,pdf}")


if __name__ == "__main__":
    main()
