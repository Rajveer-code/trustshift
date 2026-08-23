"""Round-7 scientific freeze: consolidation artifacts only (items 3, 5, 11) -- no new modeling,
no attempt to rescue H2b. Reformats/labels results already computed in rounds 5-6.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402


def _load(name):
    fp = P["out"] / name
    return json.loads(fp.read_text()) if fp.exists() else None


# ---------------------------------------------------------------------------------------------
# item 3/4: context generalization, with explicit pre-deployment-availability classification
# ---------------------------------------------------------------------------------------------
def context_generalization_final() -> dict:
    gen = _load("state_identity_generalization.json")
    axes_b = {ax: v["scheme_b_within_state_10fold"]["M1_state_identity_r2_out"] for ax, v in gen["per_axis"].items()}
    axes_a = {ax: v["scheme_a_leave_entire_state_out"]["M1_state_identity_r2_out"] for ax, v in gen["per_axis"].items()}
    out = {
        "context_variable_used": "state_fips (categorical, one-hot/fixed-effect encoded)",
        "representation": "51-level categorical dummy encoding via statsmodels C(state_fips); no "
                          "state-level continuous covariates (e.g. regulatory index, median "
                          "income, population) were available in the current lending feature set "
                          "(CATE-HMDA feature_sets.json carries state_fips and cbsa_code as IDs, "
                          "not economic/regulatory covariates) -- a bare identifier was tested, "
                          "not a richer context representation. Flagged as a scope limit, not "
                          "resolved here.",
        "framing": "Deployment context/population composition (as captured by which U.S. state a "
                  "loan-approval model is deployed in) explains substantial within-domain "
                  "heterogeneity in failure pattern that a scalar shift score (AUC_dc) does not "
                  "capture -- NOT phrased as 'state identity is predictive' without qualification.",
        "operational_availability_classification": {
            "the_identifier_itself": "AVAILABLE BEFORE DEPLOYMENT, TRIVIALLY -- you always know "
                                     "which state you are about to deploy a lending model into; "
                                     "this requires no target data or feature computation, unlike "
                                     "AUC_dc which needs target-side feature vectors.",
            "the_PREDICTIVE_VALUE_of_the_identifier": "AVAILABLE ONLY IF THAT SPECIFIC STATE HAS "
                                     "PRIOR AUDIT HISTORY -- scheme (b)'s R2=0.63-0.84 comes "
                                     "entirely from a state's OWN other (year,seed) observations "
                                     "being present in training. Scheme (a) (leave-entire-state-"
                                     "out) shows the predictive value collapses to ~0 for a state "
                                     "with ZERO prior audit history (R2_out range: "
                                     f"[{min(axes_a.values()):.3f}, {max(axes_a.values()):.3f}]).",
            "conclusion": "State identity is NOT a general, universally-applicable TrustShift "
                         "diagnostic usable on day one of a first-ever deployment to a new "
                         "population -- it is a diagnostic that requires an existing audit history "
                         "for THAT population to be useful, which is a materially different (and "
                         "narrower) kind of signal than a feature-computable probe like AUC_dc "
                         "that works on any target with available data, seen or unseen. Presented "
                         "as: 'evidence that scalar marginal shift metrics can miss structured, "
                         "learnable deployment heterogeneity, when audit history for the specific "
                         "population exists' -- not as a general pre-deployment diagnostic.",
        },
        "within_state_10fold_r2_out_by_axis": axes_b,
        "leave_entire_state_out_r2_out_by_axis": axes_a,
    }
    (P["out"] / "context_generalization_final.json").write_text(json.dumps(out, indent=2))
    print("context_generalization_final: wrote context_generalization_final.json")
    return out


# ---------------------------------------------------------------------------------------------
# item 5: magnitude-proxy sensitivity -- FINAL DECISION, no new proxies implemented
# ---------------------------------------------------------------------------------------------
def magnitude_proxy_final_decision() -> dict:
    summ = _load("magnitude_proxy_sensitivity_summary.json")
    out = {
        "proxies_tested": ["AUC_dc (domain-classifier AUC)", "|Delta-pi| (absolute prevalence shift)"],
        "proxies_NOT_implemented": {
            "PSI": "population stability index -- not computed anywhere in this codebase",
            "score_distribution_drift": "not computed as a standalone scalar anywhere in this codebase",
            "reason_not_added": "explicit user instruction: do not implement PSI/score-drift merely "
                                "to inflate the sensitivity analysis. Two proxies, honestly analyzed "
                                "with one real, explained exception, is treated as sufficient "
                                "evidence for H2a's robustness claim; the proxy analysis is marked "
                                "LIMITED (2 of 4 conceivable proxies) rather than papered over.",
        },
        "decision": "AUC_dc and |Delta-pi| are treated as sufficient for the final H2a claim.",
        "final_sentence": (
            "The conclusion that scalar shift magnitude is insufficient to characterize the "
            "observed failure fingerprint is supported by two available magnitude proxies "
            "(AUC_dc and |Delta-pi|), with one expected exception: |Delta-pi| vs. ECE (R2=0.18), "
            "which is a known, mechanistically-explained relationship (a split's calibration is "
            "judged against its own base rate, so prevalence shift mechanically affects apparent "
            "calibration error) rather than a counterexample to H2a."
        ),
        "scope_status": "LIMITED (2 of >=4 conceivable magnitude proxies tested) -- not claimed as exhaustive.",
        "source_summary": summ,
    }
    (P["out"] / "magnitude_proxy_final_decision.json").write_text(json.dumps(out, indent=2))
    print("magnitude_proxy_final_decision: wrote magnitude_proxy_final_decision.json")
    return out


# ---------------------------------------------------------------------------------------------
# item 11: aggregation-induced reversal, final consolidated schema
# ---------------------------------------------------------------------------------------------
def lending_aggregation_effect_final() -> dict:
    recon = pd.read_csv(P["out"] / "lending_f1_aggregation_reconciliation.csv")
    rows = []
    for _, r in recon.iterrows():
        rows.append({
            "year": int(r["year"]),
            "aggregate_effect": r["aggregate_delta_macro_f1"],
            "weighted_cell_effect": r["n_rows_weighted_cell_delta_macro_f1"],
            "unweighted_cell_effect": r["mean_cell_delta_macro_f1"],
            "median_cell_effect": r["median_cell_delta_macro_f1"],
            "fraction_worsening": r["fraction_cells_degrading"],
            "fraction_improving": r["fraction_cells_improving"],
            "n_cells": int(r["n_cells"]),
            "n_rows_total": int(r["n_rows_total_covered"]),
            "largest_states_by_n_rows_and_their_cell_delta": json.loads(r["largest_states_by_n_rows"]),
            "gap_weighted_vs_aggregate": r["gap_weighted_cell_vs_aggregate"],
        })
    overall_frac_worsening = float(recon["fraction_cells_degrading"].mean())
    terminology_check = {
        "majority_of_individual_cells_direction": "degrading" if overall_frac_worsening > 0.5 else "improving",
        "pooled_aggregate_direction": "improving (all 3 years negative delta)" if (recon["aggregate_delta_macro_f1"] < 0).all() else "mixed",
        "sign_reversal_confirmed": bool(overall_frac_worsening > 0.5 and (recon["aggregate_delta_macro_f1"] < 0).all()),
        "formal_simpsons_paradox_conditions_verified": False,
        "reason_not_asserted_as_formal_simpsons_paradox": "the classic two-group textbook "
            "construction and a demonstrated causal/structural lurking-variable link were not "
            "formally verified here -- only the population-size correlation with inclusion weight. "
            "The safer, precise term 'aggregation-induced reversal' (equivalently: a weighting/"
            "composition effect) is used throughout instead.",
        "mechanism": "N-weighting by target-cell row count reproduces the pooled aggregate to "
                     "within 0.005-0.008; the unweighted 447-cell mean differs by 0.024-0.043 -- "
                     "large states dominate the true aggregate while the unweighted mean gives "
                     "every state, however small, equal vote.",
    }
    out = {"by_year": rows, "terminology_check": terminology_check,
          "candidate_status": "potential secondary methodological result for Phase 4 -- not yet "
                              "written into the manuscript."}
    (P["out"] / "lending_aggregation_effect_final.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"lending_aggregation_effect_final: sign_reversal_confirmed="
          f"{terminology_check['sign_reversal_confirmed']}")
    return out


def run():
    context_generalization_final()
    magnitude_proxy_final_decision()
    lending_aggregation_effect_final()
    print("phase3_round7_freeze.run() complete")


if __name__ == "__main__":
    run()
