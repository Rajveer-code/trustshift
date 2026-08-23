"""Phase 3G global synthesis (docs/PHASE3_DESIGN.md 'Primary and secondary endpoints', T3.3 item
3): the failure-fingerprint matrix extended with severity, plus the secondary magnitude analysis
with within-domain and cross-domain relationships reported SEPARATELY and clustered appropriately
-- never one pooled omnibus regression across mechanisms (PHASE3_DESIGN.md explicitly rejects
that as the primary inferential tool).

Failure threshold: |delta| > 0.02 on the relevant axis -- the SAME tolerance already used in
synthesis/phase3_mechanism_lib.py (AUC_DC_TOL) and synthesis/phase3_negative_controls.py
(DEGRADATION_TOL), reused here rather than inventing a fourth number for the same concept.

The fingerprint matrix is built from T3.2/S1's severity ladder and T3.3's mechanism x severity
cells (one row per mechanism x domain x severity) -- NOT a re-pooling of T3.1's many (state,year)
points, which already has its own dedicated continuous-AUC_dc fingerprint in
results/lending_phase3_analysis.json (T3.1's own primary endpoint). Axes not measured for a given
mechanism (e.g. S1 never computed ECE/Brier/macro-F1 -- its mandatory reporting spec was recall
+ AUC only, PHASE3_DESIGN.md T3.2) are recorded as data_available=False, never silently treated
as "no failure".
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P, N_BOOT  # noqa: E402

FAIL_TOL = 0.02
AXES = ["auc", "macro_f1", "ece", "brier", "gap_auc"]
AXIS_NAME = {"auc": "discrimination", "macro_f1": "operating_point", "ece": "calibration",
            "brier": "calibration_brier", "gap_auc": "subgroup"}


def _load(name):
    fp = P["out"] / name
    return json.loads(fp.read_text()) if fp.exists() else None


def _fingerprint_row(source, domain, mechanism, severity, deltas: dict, available_axes: set):
    row = {"source": source, "domain": domain, "mechanism": mechanism, "severity": severity}
    for ax in AXES:
        key = f"delta_{ax}"
        if ax in available_axes and key in deltas and deltas[key] is not None:
            v = deltas[key]
            row[f"{ax}_delta"] = v
            row[f"{ax}_failed"] = bool(abs(v) > FAIL_TOL)
            row[f"{ax}_available"] = True
        else:
            row[f"{ax}_delta"] = None
            row[f"{ax}_failed"] = None
            row[f"{ax}_available"] = False
    return row


def build_fingerprint() -> tuple[pd.DataFrame, list[dict]]:
    """Only FEASIBLE T3.3 severity cells enter the fingerprint -- an infeasible construction
    (PHASE3_DESIGN.md's own failure-condition column) is not evidence about the mechanism, it is
    evidence the construction couldn't be built at that severity/stratum. Every infeasible cell
    is still logged (never silently dropped), mirroring T3.1's lending_excluded_cells.json
    pattern, so the exclusion is auditable."""
    rows, excluded = [], []

    def _add(source, domain, mechanism, level, d, axes):
        if "feasible" in d and not d["feasible"]:
            excluded.append({"source": source, "domain": domain, "mechanism": mechanism,
                            "severity": level, "reason": "feasible=False",
                            "reaches_target": d.get("reaches_target"),
                            "replacement_frac": d.get("replacement_frac"),
                            "achieved_auc_dc": d.get("achieved_auc_dc"),
                            "target_auc_dc": d.get("target_auc_dc")})
            return
        rows.append(_fingerprint_row(source, domain, mechanism, level, d, axes))

    lending = _load("phase3_T33_lending.json")
    security = _load("phase3_T33_security.json")
    s1 = _load("phase3_T32_S1_full.json")

    if lending:
        cov = lending["cells"]["covariate"]
        for level, d in cov.get("severities", {}).items():
            _add("T3.3", "lending", "covariate", level, d, {"auc", "macro_f1", "ece", "brier", "gap_auc"})
        pri = lending["cells"]["prior"]
        for level, d in pri.get("severities", {}).items():
            _add("T3.3", "lending", "prior", level, d, {"auc", "macro_f1", "ece", "brier", "gap_auc"})
        con = lending["cells"]["concept_natural"]
        if con.get("status") == "FOUND":
            d = {"delta_auc": con["delta_auc"], "delta_macro_f1": con["delta_macro_f1"],
                "delta_ece": con["actual_delta_ece"], "delta_brier": con["delta_brier"],
                "delta_gap_auc": con["delta_gap_auc"]}
            rows.append(_fingerprint_row("T3.3", "lending", "concept_natural", "discovered", d,
                                         {"auc", "macro_f1", "ece", "brier", "gap_auc"}))

    if security:
        cov = security["cells"]["covariate"]
        if cov.get("status") == "OK":
            for level, d in cov.get("severities", {}).items():
                _add("T3.3", "security", "covariate", level, d, {"auc", "macro_f1", "ece", "brier"})
        pri = security["cells"]["prior"]
        if pri.get("status") == "OK":
            for level, d in pri.get("severities", {}).items():
                _add("T3.3", "security", "prior", level, d, {"auc", "macro_f1", "ece", "brier"})

    if s1:
        baseline_auc = {p["seed"]: p["target_cicids2017"]["auc"] for p in s1["baseline"]["per_seed"]}
        for level, entry in s1.get("severity_levels", {}).items():
            aucs = [p["target_cicids2017"]["auc"] - baseline_auc.get(p["seed"], np.nan)
                   for p in entry["per_seed"]]
            aucs = [a for a in aucs if a == a]
            recalls = [p["target_cicids2017"]["overall_attack_recall"]["recall"] for p in entry["per_seed"]
                      if p["target_cicids2017"]["overall_attack_recall"]["recall"] is not None]
            if aucs:
                d = {"delta_auc": float(np.mean(aucs))}
                r = _fingerprint_row("T3.2/S1", "security", "novel_mechanism", level, d, {"auc"})
                r["overall_attack_recall_mean"] = float(np.mean(recalls)) if recalls else None
                rows.append(r)

    return pd.DataFrame(rows), excluded


def _ols_with_seed_bootstrap(x, y, groups, n_boot=N_BOOT, seed=42):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0:
        return {"slope": None, "slope_ci95": [None, None], "r2": None, "n": len(x)}
    b1, b0 = np.polyfit(x, y, 1)
    yhat = b1 * x + b0
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        rs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in rs])
        if np.std(x[idx]) == 0:
            continue
        s, _ = np.polyfit(x[idx], y[idx], 1)
        slopes.append(s)
    lo, hi = (float(v) for v in np.percentile(slopes, [2.5, 97.5])) if slopes else (None, None)
    return {"slope": float(b1), "slope_ci95": [lo, hi], "r2": r2, "n": len(x)}


def magnitude_analysis() -> dict:
    out = {"within_domain": {}, "cross_domain": {}, "note": (
        "Within-domain and cross-domain relationships are reported SEPARATELY and clustered "
        "appropriately -- PHASE3_DESIGN.md explicitly rejects a single pooled omnibus regression "
        "across mechanisms/domains as the primary inferential tool (Simpson-style clustering risk, "
        "the exact failure mode that already invalidated the original cross-domain M2 slope)."
    )}

    lending_fp = _load("lending_phase3_analysis.json")
    if lending_fp:
        ax = lending_fp["fingerprint"]["axes"]["delta_gap_auc"]
        out["within_domain"]["lending_T31_covariate_like"] = {
            "mechanism": "covariate_like (geographic/temporal, real-world)",
            "x": "continuous auc_dc", "y": "delta_gap_auc",
            "ols_slope": ax["ols_slope"], "ols_slope_ci95": ax["ols_slope_ci95"], "r2": ax["r2"],
            "classification": ax["classification"], "n_cells": lending_fp["n_cells"],
            "clustering": "cluster (block) bootstrap over states -- see lending_phase3_analysis.json",
        }

    s1 = _load("phase3_T32_S1_full.json")
    if s1:
        x, y_auc, y_recall, seeds = [], [], [], []
        baseline_auc = {p["seed"]: p["target_cicids2017"]["auc"] for p in s1["baseline"]["per_seed"]}
        baseline_recall = {p["seed"]: p["target_cicids2017"]["overall_attack_recall"]["recall"]
                           for p in s1["baseline"]["per_seed"]}
        for level, entry in s1["severity_levels"].items():
            for p in entry["per_seed"]:
                x.append(entry["coverage_reduction_frac"])
                y_auc.append(baseline_auc[p["seed"]] - p["target_cicids2017"]["auc"])
                y_recall.append(baseline_recall[p["seed"]] - p["target_cicids2017"]["overall_attack_recall"]["recall"])
                seeds.append(p["seed"])
        fit_auc = _ols_with_seed_bootstrap(x, y_auc, np.array(seeds))
        fit_recall = _ols_with_seed_bootstrap(x, y_recall, np.array(seeds))
        clustering_note = ("bootstrap over the 3 model seeds (only 3 severity levels x 3 seeds = "
                           "9 points -- explicitly a thin-data caveat, not a cluster-bootstrap-"
                           "over-states like T3.1)")
        out["within_domain"]["security_S1_novel_mechanism_auc"] = {
            "mechanism": "novel_mechanism (mechanism-coverage holdout)",
            "x": "coverage_reduction_frac (0.2/0.4/0.6)", "y": "delta_auc (target_cicids2017)",
            **fit_auc, "clustering": clustering_note}
        out["within_domain"]["security_S1_novel_mechanism_recall"] = {
            "mechanism": "novel_mechanism (mechanism-coverage holdout)",
            "x": "coverage_reduction_frac (0.2/0.4/0.6)",
            "y": "delta_overall_attack_recall (target_cicids2017) -- the design doc's own "
                "MANDATORY S1 reporting metric, less noisy than AUC here (see per-seed std in "
                "security_mechanism_ladder.csv)",
            **fit_recall, "clustering": clustering_note,
            "note": ("Target overall_attack_recall is ~0.145-0.146 across EVERY condition "
                     "(baseline, every holdout severity, every matched negative-control severity) "
                     "-- essentially flat. Cross-dataset recall failure is already near-total at "
                     "baseline (0 families held out) and does not scale with in-source "
                     "mechanism-coverage severity; something other than which specific "
                     "reflection_volumetric families were trained on dominates the cross-testbed "
                     "gap. This is an unexpected result relative to h2's stated expectation and is "
                     "reported plainly, not smoothed over -- see security_mechanism_ladder.csv."),
        }

    fp_df, _ = build_fingerprint()
    cov = fp_df[(fp_df.mechanism == "covariate") & fp_df.auc_available]
    if len(cov) >= 3:
        sev_map = {"low": 0.60, "medium": 0.75, "high": 0.90}
        x = cov["severity"].map(sev_map).to_numpy(dtype=float)
        y = cov["auc_delta"].to_numpy(dtype=float)
        domains_arr = cov["domain"].to_numpy()
        valid = ~np.isnan(x)
        fit = _ols_with_seed_bootstrap(x[valid], y[valid], domains_arr[valid])
        out["cross_domain"]["covariate_mechanism_only"] = {
            "scope": "covariate mechanism ONLY -- the sole mechanism with a common continuous "
                    "magnitude scale (target auc_dc) across domains; prior uses domain-specific "
                    "|Delta-pi| targets and novel-mechanism uses coverage_reduction_frac, neither "
                    "of which is comparable across domains on one axis, so no cross-domain number "
                    "is computed for them.",
            "x": "target_auc_dc (0.60/0.75/0.90)", "y": "delta_auc",
            "domains_included": sorted(cov["domain"].unique().tolist()), **fit,
            "clustering": "bootstrap over domain (lending vs security) -- only 2 clusters, "
                         "explicitly thin; interpret as descriptive, not a hypothesis test.",
        }
    else:
        out["cross_domain"]["covariate_mechanism_only"] = {
            "status": "INSUFFICIENT_DATA", "n_available": int(len(cov))}

    return out


def run():
    fp_df, excluded = build_fingerprint()
    fp_df.to_csv(P["out"] / "phase3_failure_fingerprint.csv", index=False)
    (P["out"] / "phase3_T33_infeasible_constructions.json").write_text(json.dumps({
        "n_excluded": len(excluded),
        "note": "T3.3 severity cells with feasible=False (PHASE3_DESIGN.md's own failure-"
                "condition column) -- excluded from the fingerprint, logged here so no "
                "infeasible construction is silently dropped without a reason.",
        "excluded": excluded,
    }, indent=2))
    mag = magnitude_analysis()
    (P["out"] / "phase3_magnitude_analysis.json").write_text(json.dumps(mag, indent=2))
    print(f"wrote phase3_failure_fingerprint.csv ({len(fp_df)} rows, {len(excluded)} infeasible "
          f"cells excluded and logged), phase3_magnitude_analysis.json")
    return fp_df, mag


if __name__ == "__main__":
    run()
