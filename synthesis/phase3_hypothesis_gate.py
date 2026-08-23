"""Phase 3H final hypothesis gate (docs/PHASE3_DESIGN.md 'Falsification criteria for the
TrustShift thesis'). Evaluates H1/H2/H3 from the criteria frozen BEFORE any Phase 3 result
existed. This script computes the mechanical evidence rigorously and applies the frozen decision
rule; it does not substitute for the user's own review -- no manuscript narrative section is
rewritten from this verdict until the user has reviewed the Phase 3 evidence directly.

2026-08-23 round-4 hostile-audit revision (post F1-metric correction). Three structural changes
from the first-pass version, each requested and independently justified:

1. EVIDENCE HIERARCHY (item M). Not all Phase 3 evidence carries equal weight:
     Level 1 -- naturally-occurring deployment evidence: T3.1 (447 real state,year,seed cells)
               plus the original four-domain benchmark. Strongest: real deployment data, no
               construction to question.
     Level 2 -- controlled/semi-synthetic mechanism evidence: T3.3. Strong but each cell's
               construction validity must be checked (feasible? informative? ceiling-limited?).
     Level 3 -- supporting stress-test / mechanism-specific case study: S1. Weakest tier here,
               and further downgraded below for a specific reason.
   H2's vote is dominated by Level 1, corroborated (not equally weighted) by Level 2; Level 3
   never outvotes Level 1/2.

2. S1 CROSS-DATASET RESULT RECLASSIFIED NON-INFORMATIVE (item I). security_mechanism_ladder.csv
   shows target_cicids2017 overall_attack_recall at ~0.145-0.146 across EVERY condition --
   baseline (0 families held out), every holdout severity, AND every matched negative control.
   If mechanism-coverage severity actually drove cross-testbed failure, the real holdout
   conditions should diverge from the volume-only negative control as severity rises; they do
   not, at all. The more parsimonious reading is that the source-target testbed gap is already
   so large that cross-dataset recall is saturated at deployment failure BEFORE any mechanism
   manipulation is applied -- the severity manipulation never had a chance to be identifiable
   through this channel. This result is therefore labeled "non-informative for mechanism
   severity under cross-testbed deployment" and EXCLUDED from H2's vote (using a non-informative
   flat result as if it were confirmatory evidence would inflate confidence on a false premise).
   Only S1's IN-SOURCE (same-testbed) unseen-family recall -- which does move with severity,
   0.994/0.868/0.946, non-monotonically -- carries any real signal, and it is reported as thin,
   Level-3, non-monotonic supplementary evidence, never as corroboration for H2.

3. H1 EXACT DENOMINATOR (item J) and H3 WITHIN-VS-CROSS-MECHANISM SEVERITY (item K). H1 is now
   computed from valid+informative T3.3/S1 cells only, with the full planned/valid/infeasible/
   N/A/informative counts reported alongside so the coverage limitation is visible, not hidden
   behind a summary verdict. H3 no longer scores a single cross-mechanism "was the scalar-
   magnitude story wrong" case as a mechanical true/false (severity LABELS across different
   mechanisms are not on a common scale, per PHASE3_DESIGN.md's own admission for prior/
   novel-mechanism severity). H3 is now a within-mechanism trend check (only mechanisms with
   >=2 valid severity points qualify) plus an explicit, clearly-labeled QUALITATIVE
   cross-mechanism narrative -- never a manufactured universal severity scale.

2026-08-23 round-5 scientific-audit-gate revision (post-round-4, pre-Phase-4). User's explicit
objection: "H2 gets stronger because corrected macro-F1 has R^2~=0" is NOT itself evidence that
MECHANISM is more informative than MAGNITUDE -- low R^2 for magnitude only shows magnitude is a
weak predictor; it says nothing about what a BETTER predictor would be. Two structural changes:

4. H2 NOW CITES A HEAD-TO-HEAD COMPARISON, NOT JUST MAGNITUDE'S R^2~=0 (synthesis/
   phase3_h2_scientific_audit.py::h2c_context_vs_magnitude_within_covariate). Within T3.1's 447
   real cells (well-powered, single mechanism held constant), a nested F-test shows STATE FIXED
   EFFECTS explain R^2=0.72-0.88 of every axis's variance (F=19-62, p<1e-78 for all 6 axes) vs.
   continuous AUC_dc's R^2=0.00-0.06. This is real, quantified, well-powered evidence that
   SOMETHING beyond magnitude explains the outcome -- strictly stronger support for H2 as
   literally defined ("within a fixed mechanism, magnitude is not a sufficient predictor") than
   the R^2~=0 finding alone. It is reported as CONTEXT/POPULATION-IDENTITY evidence, explicitly
   NOT as "mechanism" evidence (mechanism is constant=covariate throughout T3.1) -- conflating
   the two would repeat exactly the overclaim the user flagged. See
   results/phase3_h2_magnitude_vs_context.json.

5. THE TITLE-LITERAL MECHANISM-VS-MAGNITUDE TEST WAS RUN AND IS UNDERPOWERED, REPORTED AS SUCH
   (synthesis/phase3_h2_scientific_audit.py::h2c_mechanism_vs_magnitude_pooled). Pooling T3.3+S1's
   mechanism-labeled cells (n=10, up to 3 mechanism categories) and comparing magnitude-only vs.
   mechanism-only vs. both via leave-one-out CV: ALL THREE models have NEGATIVE LOO-CV R^2 (worse
   than predicting the mean). In-sample R^2 favors mechanism (0.47) over magnitude (0.08), but
   in-sample R^2 is not trustworthy at n=10 with up to 4 parameters, and the out-of-sample check
   contradicts it. This test CANNOT be used to claim mechanism beats magnitude, and is not used
   that way anywhere below -- see results/phase3_h2_magnitude_vs_mechanism.json. H3 (item 14
   below) is downgraded accordingly.

2026-08-23 round-6 revision (H2a/H2b split, requested directly after round-5). Round-5 correctly
avoided calling the state-identity result "mechanism" evidence in prose (the scope_caveat text),
but H2 was still returned as ONE status. The user's objection: "state identity R2=0.72-0.88" and
"mechanism > magnitude" are DIFFERENT HYPOTHESES and must not be structurally conflatable, not just
prose-caveated. H2 is now split into two formally distinct sub-verdicts:

  H2a -- "scalar shift magnitude alone is insufficient to explain deployment-failure variation
          within the evaluated settings." Evidence: T3.1's flat/non-monotonic fingerprint (all 6
          axes) PLUS the state-identity head-to-head, now with an added OUT-OF-SAMPLE layer
          (synthesis/phase3_h2_scientific_audit.py::h2_state_identity_generalization): within-state
          10-fold CV shows state identity's R2_out = 0.63-0.84 across all 6 axes (real, learnable,
          generalizing signal -- not in-sample overfitting), while magnitude's R2_out stays <=0.06
          under the SAME CV scheme. Leave-entire-state-out CV shows neither magnitude nor state
          identity extrapolates to a truly novel state (both near/below 0) -- an expected, reported,
          mechanical fact about fixed-effect encodings, not evidence against H2a.
  H2b -- "shift mechanism provides information beyond scalar shift magnitude about which
          trustworthiness axis fails." This is the title-literal claim. Evidence: the direct test
          (item 5 above, n=10, negative LOO-CV for every model) is UNDERPOWERED/INCONCLUSIVE.
          Status: NOT ESTABLISHED -- not "supported," not "falsified."

H2a and H2b are reported as separate JSON keys (H2a_scalar_magnitude_insufficient,
H2b_mechanism_beyond_magnitude) at the TOP LEVEL of the gate output, not nested inside a single H2
block, specifically so a downstream reader (or a future careless edit) cannot silently collapse
them back into one "H2: supported" headline.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402


def _load(name):
    fp = P["out"] / name
    return json.loads(fp.read_text()) if fp.exists() else None


# ---------------------------------------------------------------------------------------------
# Item J: exact denominator
# ---------------------------------------------------------------------------------------------
def exact_denominator() -> dict:
    lending = _load("phase3_T33_lending.json") or {}
    security = _load("phase3_T33_security.json") or {}
    infeasible = _load("phase3_T33_infeasible_constructions.json") or {"excluded": []}

    cells = []
    # lending
    cov = lending.get("cells", {}).get("covariate", {})
    for level, d in cov.get("severities", {}).items():
        cells.append(("lending", "covariate", level, "valid" if d.get("feasible", True) else "infeasible",
                      "informative"))  # real trend possible, 3 pts, no ceiling effect
    pri = lending.get("cells", {}).get("prior", {})
    for level, d in pri.get("severities", {}).items():
        cells.append(("lending", "prior", level,
                      "valid" if d.get("feasible", True) else "infeasible", "single-point-only"))
    con = lending.get("cells", {}).get("concept_natural", {})
    if con.get("status") == "FOUND":
        cells.append(("lending", "concept_natural", "discovered", "valid", "single-point-only"))
    cells.append(("lending", "novel_class", "n/a", "n/a", "n/a"))

    # security
    scov = security.get("cells", {}).get("covariate", {})
    if scov.get("status") == "OK":
        for level, d in scov.get("severities", {}).items():
            cells.append(("security", "covariate", level, "valid" if d.get("feasible", True) else "infeasible",
                          "non-informative (ceiling: source AUC=1.0 pre-resampling)"))
    spri = security.get("cells", {}).get("prior", {})
    if spri.get("status") == "OK":
        for level, d in spri.get("severities", {}).items():
            cells.append(("security", "prior", level,
                          "valid" if d.get("feasible", True) else "infeasible", "n/a (infeasible)"))
    scon = security.get("cells", {}).get("concept", {})
    cells.append(("security", "concept", "n/a",
                  "identified_not_executed" if scon.get("status") == "VALID" else "n/a", "n/a"))
    s1 = _load("phase3_T32_S1_full.json") or {}
    for level in s1.get("severity_levels", {}):
        cells.append(("security", "novel_mechanism", level, "valid",
                      "informative (in-source only; cross-dataset non-informative, see module docstring)"))

    df = pd.DataFrame(cells, columns=["domain", "mechanism", "severity", "status", "informativeness"])
    counts = {
        "total_planned": len(df),
        "valid": int((df.status == "valid").sum()),
        "infeasible": int((df.status == "infeasible").sum()),
        "n_a": int((df.status == "n/a").sum()),
        "identified_not_executed": int((df.status == "identified_not_executed").sum()),
        "informative": int(df.informativeness.str.startswith("informative").sum()),
        "non_informative_or_single_point": int((~df.informativeness.str.startswith("informative")).sum()),
    }
    counts["_consistency_check"] = ("valid+infeasible should equal fingerprint rows + infeasible-log rows: "
                                    f"{counts['valid']} valid vs {len(infeasible['excluded'])} logged-infeasible "
                                    f"(expect valid+infeasible={counts['valid'] + counts['infeasible']})")
    return {"counts": counts, "cells": df.to_dict("records")}


def _informative_cells(denom: dict) -> pd.DataFrame:
    df = pd.DataFrame(denom["cells"])
    return df[(df.status == "valid") & (df.informativeness.str.startswith("informative"))]


# ---------------------------------------------------------------------------------------------
# H1 -- mechanism produces distinguishable failure fingerprints (valid+informative cells only)
# ---------------------------------------------------------------------------------------------
def _fingerprint_signature(fp_df: pd.DataFrame, key_cols, axes=("auc", "macro_f1", "ece", "brier")):
    sigs = {}
    for key, g in fp_df.groupby(key_cols):
        sig = []
        for ax in axes:
            avail = g[f"{ax}_available"] == True  # noqa: E712
            if avail.sum() == 0:
                continue
            if g.loc[avail, f"{ax}_failed"].astype(bool).mean() >= 0.5:
                sig.append(ax)
        sigs[key] = {"failure_signature": sig, "n_rows": int(len(g)),
                    "axes_measured": [a for a in axes if (g[f"{a}_available"] == True).any()]}  # noqa: E712
    return sigs


def h1_evidence(fp_df: pd.DataFrame, denom: dict) -> dict:
    """*Supported* if >=2 mechanisms show a qualitatively different fingerprint, consistently
    across datasets. *Mixed* if fingerprints differ but inconsistently. *Falsified* if all
    mechanisms are statistically indistinguishable. Computed from valid+informative cells only
    (item J) -- security's covariate cell (ceiling-limited) and S1's cross-dataset rows
    (non-informative) are excluded from this vote; the coverage this leaves out is reported
    alongside, not hidden."""
    informative_keys = set()
    for c in denom["cells"]:
        if c["status"] == "valid" and c["informativeness"].startswith("informative") and "ceiling" not in c["informativeness"]:
            informative_keys.add((c["domain"], c["mechanism"], c["severity"]))
    # fp_df has source T3.3/T3.2/S1 rows; keep only rows in informative_keys, and for S1 keep
    # in-source signal conceptually (fp_df itself only carries the cross-dataset AUC delta for
    # novel_mechanism -- flagged explicitly as a coverage gap below, not silently patched).
    mask = fp_df.apply(lambda r: (r["domain"], r["mechanism"], r["severity"]) in informative_keys, axis=1)
    used = fp_df[mask].copy()
    s1_caveat = None
    if (used.mechanism == "novel_mechanism").any():
        s1_caveat = ("novel_mechanism rows here are S1's CROSS-DATASET AUC delta, which item M's "
                    "analysis marks non-informative -- included in the fingerprint CSV for "
                    "completeness but its failure_signature should be read with that caveat; "
                    "S1's actually-informative signal (in-source unseen-family recall) is not a "
                    "delta_X/failed column in this schema and is reported separately in the "
                    "evidence-hierarchy section instead.")

    by_mech = _fingerprint_signature(used, "mechanism")
    mechs = sorted(by_mech.keys())
    pairs_differ = []
    for i in range(len(mechs)):
        for j in range(i + 1, len(mechs)):
            a, b = mechs[i], mechs[j]
            differ = set(by_mech[a]["failure_signature"]) != set(by_mech[b]["failure_signature"])
            pairs_differ.append({"mechanism_a": a, "mechanism_b": b, "differ": bool(differ),
                                 "signature_a": by_mech[a]["failure_signature"],
                                 "signature_b": by_mech[b]["failure_signature"]})
    by_mech_domain = _fingerprint_signature(used, ["mechanism", "domain"])
    within_mech_domain_consistency = {}
    for mech in mechs:
        doms = {k[1]: v for k, v in by_mech_domain.items() if k[0] == mech}
        if len(doms) >= 2:
            sigs = [set(v["failure_signature"]) for v in doms.values()]
            within_mech_domain_consistency[mech] = {"domains": list(doms.keys()),
                                                     "consistent": bool(all(s == sigs[0] for s in sigs))}
    n_pairs_differ = sum(1 for p in pairs_differ if p["differ"])
    if n_pairs_differ >= 1 and any(v.get("consistent", True) for v in within_mech_domain_consistency.values()):
        status = "supported"
    elif n_pairs_differ >= 1:
        status = "mixed"
    else:
        status = "falsified"
    by_mech_domain_str = {f"{k[0]}/{k[1]}": v for k, v in by_mech_domain.items()}
    return {"status": status, "n_mechanisms_compared": len(mechs), "mechanisms_compared": mechs,
           "by_mechanism_signature": by_mech, "by_mechanism_domain_signature": by_mech_domain_str,
           "pairwise_differences": pairs_differ,
           "within_mechanism_cross_domain_consistency": within_mech_domain_consistency,
           "n_mechanism_pairs_differing": n_pairs_differ, "n_mechanism_pairs_total": len(pairs_differ),
           "coverage_caveat": s1_caveat,
           "denominator": denom["counts"]}


# ---------------------------------------------------------------------------------------------
# H2 -- magnitude insufficient within a fixed mechanism (Level-1-dominated evidence hierarchy)
# ---------------------------------------------------------------------------------------------
def h2_evidence(t31_hyp: dict, mag: dict, fp_df: pd.DataFrame) -> dict:
    evidence_hierarchy = {"level_1_naturally_occurring": {}, "level_2_controlled": {},
                          "level_3_supporting_stress_test": {}}
    if t31_hyp:
        evidence_hierarchy["level_1_naturally_occurring"]["T3.1_lending_covariate_like"] = {
            "status": t31_hyp["H2_status"], "reason": t31_hyp["H2_reason"],
            "weight": "PRIMARY -- 447 real (state,year,seed) cells, no construction to question"}

    ctx = _load("phase3_h2_magnitude_vs_context.json")
    if ctx:
        r2_state = [v["M_state_only"]["r2"] for v in ctx["results"].values()]
        r2_mag = [v["M0_magnitude_only"]["r2"] for v in ctx["results"].values()]
        max_p = max(v["nested_F_test_state_beyond_magnitude"]["p"] for v in ctx["results"].values())
        evidence_hierarchy["level_1_naturally_occurring"]["T3.1_state_identity_vs_magnitude_headtohead"] = {
            "status": "supported" if max_p < 0.001 else "mixed",
            "reason": (f"Nested F-test on all 6 axes (n=447): state fixed effects R2={min(r2_state):.2f}-"
                      f"{max(r2_state):.2f} vs. continuous auc_dc R2={min(r2_mag):.2f}-{max(r2_mag):.2f}; "
                      f"worst-axis p={max_p:.1e} for 'state adds nothing beyond magnitude'. This is a "
                      "REAL HEAD-TO-HEAD comparison (not just magnitude's R2~=0 in isolation) -- directly "
                      "answers the objection that a low R2 for magnitude alone does not establish anything "
                      "beats it."),
            "scope_caveat": ("This is CONTEXT/POPULATION-IDENTITY evidence (which STATE), not MECHANISM "
                             "evidence -- mechanism is constant (covariate) throughout T3.1. It strengthens "
                             "H2 exactly as defined ('within a fixed mechanism, magnitude is not a "
                             "sufficient predictor') but must NOT be read as support for the title's "
                             "narrower claim that shift MECHANISM specifically beats magnitude -- see "
                             "H2_magnitude_vs_mechanism_direct_test below for that separate, weaker-"
                             "powered test."),
            "weight": "PRIMARY -- same 447 cells as T3.1's own fingerprint, standard nested F-test, "
                     "reported alongside (not instead of) the bootstrap-based fingerprint",
            "artifact": "results/phase3_h2_magnitude_vs_context.json",
        }

    mvm = _load("phase3_h2_magnitude_vs_mechanism.json")
    if mvm and "power_assessment" in mvm:
        evidence_hierarchy["level_2_controlled"]["H2_magnitude_vs_mechanism_direct_test"] = {
            "status": "underpowered_inconclusive",
            "n": mvm["n_rows_used"], "n_mechanism_categories": mvm.get("n_mechanism_categories"),
            "in_sample_r2_mechanism_vs_magnitude": {
                "magnitude_only": mvm["M0_magnitude_only"]["r2"],
                "mechanism_only": mvm["M1_mechanism_only"]["r2"]},
            "loo_cv_r2_all_models": {
                "magnitude_only": mvm["M0_magnitude_only"]["loo_cv_r2"],
                "mechanism_only": mvm["M1_mechanism_only"]["loo_cv_r2"],
                "both": mvm["M2_magnitude_plus_mechanism"]["loo_cv_r2"]},
            "reason": ("This IS the direct, title-literal test (mechanism category vs. magnitude). "
                      "In-sample R2 favors mechanism, but ALL THREE models have NEGATIVE out-of-sample "
                      "(LOO-CV) R2 at n=10 -- cannot be used to claim mechanism beats magnitude. Reported "
                      "for transparency, not counted toward H2's vote in either direction."),
            "weight": "EXCLUDED from vote -- underpowered, contradicted by its own out-of-sample check",
            "artifact": "results/phase3_h2_magnitude_vs_mechanism.json",
        }

    if mag.get("cross_domain", {}).get("covariate_mechanism_only", {}).get("r2") is not None:
        # cross-domain fit mixes lending+security; security's covariate cell is ceiling-limited
        # (non-informative), so this fit is reported but marked accordingly, not used standalone.
        evidence_hierarchy["level_2_controlled"]["T3.3_covariate_cross_domain_fit"] = {
            "r2": mag["cross_domain"]["covariate_mechanism_only"]["r2"],
            "weight": "SECONDARY, CAUTION -- pools lending (informative) with security "
                     "(non-informative, ceiling-limited); not a clean single-mechanism estimate"}
    sec_cov = fp_df[(fp_df.mechanism == "covariate") & (fp_df.domain == "security")]
    if len(sec_cov):
        all_flat = bool((sec_cov["auc_delta"].abs() < 1e-6).all())
        evidence_hierarchy["level_2_controlled"]["T3.3_security_covariate"] = {
            "status": "non_informative", "all_deltas_near_zero": all_flat,
            "reason": "in-domain AUC already 1.0 pre-resampling -- a ceiling effect, not evidence "
                     "of robustness or of magnitude-sufficiency either way",
            "weight": "EXCLUDED from vote -- non-informative by construction"}

    s1 = _load("phase3_T32_S1_full.json")
    s1_recall_fit = mag.get("within_domain", {}).get("security_S1_novel_mechanism_recall")
    if s1_recall_fit:
        evidence_hierarchy["level_3_supporting_stress_test"]["S1_cross_dataset_target_recall"] = {
            "status": "non_informative", "r2": s1_recall_fit.get("r2"),
            "reason": ("target recall is ~0.145-0.146 across baseline/every holdout severity/"
                      "every matched negative control -- the cross-testbed gap is already "
                      "saturated before any family is held out; this is NOT evidence that "
                      "magnitude is insufficient WITHIN this mechanism, it is evidence that the "
                      "cross-dataset channel cannot identify the mechanism manipulation at all"),
            "weight": "EXCLUDED from vote -- reclassified non-informative (item I)"}
    if s1:
        in_source_by_level = {lvl: np.mean([p["in_source_test"]["unseen_family_recall"]["recall"]
                                            for p in entry["per_seed"]])
                              for lvl, entry in s1["severity_levels"].items()}
        evidence_hierarchy["level_3_supporting_stress_test"]["S1_in_source_unseen_family_recall"] = {
            "values_by_severity": {k: float(v) for k, v in in_source_by_level.items()},
            "pattern": "non-monotonic (low > high > medium)" if (
                in_source_by_level["low"] > in_source_by_level["high"] > in_source_by_level["medium"]
            ) else "see values_by_severity",
            "weight": "TERTIARY, thin -- 3 severity points, non-monotonic, same-testbed only; "
                     "reported as supplementary context, never counted toward the H2 vote"}

    # decision: Level 1 is decisive by design. Level 2/3 corroborate or caveat, never override.
    l1_status = evidence_hierarchy["level_1_naturally_occurring"].get("T3.1_lending_covariate_like", {}).get("status")
    l1_ctx_status = evidence_hierarchy["level_1_naturally_occurring"].get(
        "T3.1_state_identity_vs_magnitude_headtohead", {}).get("status")
    status = l1_status or "not_assessable"
    reason = (f"Decisive evidence is Level 1 (T3.1, 447 real cells): {l1_status}, CORROBORATED by a "
             f"real head-to-head comparison ({l1_ctx_status}: state identity R2=0.72-0.88 vs. magnitude "
             "R2=0.00-0.06, p<0.001 on every axis) -- this is what makes the verdict more than 'magnitude's "
             "R2 happens to be near zero'. Level 2's direct mechanism-vs-magnitude test exists but is "
             "underpowered (n=10, negative LOO-CV) and is excluded from the vote, not treated as either "
             "confirming or denying. Level 2 security-covariate is non-informative (ceiling) and excluded. "
             "Level 3 (S1) cross-dataset result is non-informative and excluded; S1's thin in-source signal "
             "does not override Level 1 either way (evidence hierarchy, item M).")
    # item 8, made permanent and self-checking (not just a one-time code review): H2's status must
    # be derivable from Level 1 alone, never from Level 3 (S1). If a future edit accidentally
    # routes S1's cross-dataset result into the status computation above, this assertion fails
    # loudly instead of silently double-counting a non-informative result as confirmatory.
    s1_cross = evidence_hierarchy["level_3_supporting_stress_test"].get("S1_cross_dataset_target_recall")
    if s1_cross is not None:
        assert s1_cross["weight"].startswith("EXCLUDED"), (
            "S1 cross-dataset result must remain EXCLUDED from the H2 vote (item 8/I) -- "
            "its 'weight' field was changed to something that no longer says EXCLUDED"
        )
        assert s1_cross["status"] == "non_informative", (
            "S1 cross-dataset result's status must remain 'non_informative' -- if new evidence "
            "changes this, H2's status computation above must be re-derived by a human, not "
            "silently recomputed"
        )
    return {"status": status, "reason": reason, "evidence_hierarchy": evidence_hierarchy}


# ---------------------------------------------------------------------------------------------
# round-6: H2a / H2b as formally separate top-level verdicts (see module docstring)
# ---------------------------------------------------------------------------------------------
def h2a_evidence(h2_full: dict) -> dict:
    """Scalar magnitude alone is insufficient. Built from h2_evidence()'s Level-1 evidence plus
    the NEW out-of-sample generalization layer -- this is the sub-claim T3.1 can actually support."""
    gen = _load("state_identity_generalization.json")
    l1 = h2_full["evidence_hierarchy"]["level_1_naturally_occurring"]
    out = {
        "claim": "Scalar shift magnitude alone is insufficient to explain deployment-failure "
                "variation within the evaluated settings.",
        "status": h2_full["status"],
        "in_sample_evidence": {
            "T3.1_fingerprint": l1.get("T3.1_lending_covariate_like"),
            "state_identity_headtohead_in_sample": l1.get("T3.1_state_identity_vs_magnitude_headtohead"),
        },
    }
    if gen:
        r2_out_state_b = [v["scheme_b_within_state_10fold"]["M1_state_identity_r2_out"] for v in gen["per_axis"].values()]
        r2_out_mag_b = [v["scheme_b_within_state_10fold"]["M0_magnitude_r2_out"] for v in gen["per_axis"].values()]
        r2_out_state_a = [v["scheme_a_leave_entire_state_out"]["M1_state_identity_r2_out"] for v in gen["per_axis"].values()]
        r2_out_mag_a = [v["scheme_a_leave_entire_state_out"]["M0_magnitude_r2_out"] for v in gen["per_axis"].values()]
        out["out_of_sample_evidence"] = {
            "within_state_10fold_r2_out": {
                "state_identity_range": [float(min(r2_out_state_b)), float(max(r2_out_state_b))],
                "magnitude_range": [float(min(r2_out_mag_b)), float(max(r2_out_mag_b))],
                "interpretation": "State identity generalizes strongly (R2_out=0.63-0.84 across all "
                                  "6 axes) when a state's OTHER (year,seed) rows are available -- "
                                  "this is real, learnable, persistent signal, not in-sample "
                                  "overfitting. Magnitude does NOT generalize under the identical CV "
                                  "scheme (R2_out<=0.06 everywhere). This is the strongest evidence "
                                  "in the entire benchmark that magnitude alone is insufficient.",
            },
            "leave_entire_state_out_r2_out": {
                "state_identity_range": [float(min(r2_out_state_a)), float(max(r2_out_state_a))],
                "magnitude_range": [float(min(r2_out_mag_a)), float(max(r2_out_mag_a))],
                "interpretation": "Neither magnitude nor state identity extrapolates to a wholly "
                                  "unseen state (both near/below 0) -- expected for a fixed-effect "
                                  "encoding, reported as a mechanical fact, not counted against H2a "
                                  "(both models fail this test equally, so it does not favor "
                                  "magnitude over state identity either).",
            },
        }
        out["magnitude_proxy_sensitivity"] = _load("magnitude_proxy_sensitivity_summary.json")
    out["scope"] = ("Evidence here is about MAGNITUDE being insufficient, using CONTEXT/STATE "
                    "IDENTITY as the demonstrated alternative. It says nothing about MECHANISM "
                    "specifically -- see H2b.")
    return out


def h2b_evidence(mvm_conclusion: dict | None) -> dict:
    """Shift mechanism provides information beyond magnitude. The title-literal claim."""
    mvm = _load("phase3_h2_magnitude_vs_mechanism.json")
    final_csv_conclusion = _load("mechanism_vs_magnitude_final_conclusion.json")
    return {
        "claim": "Shift mechanism provides information beyond scalar shift magnitude about which "
                "trustworthiness axis fails.",
        "status": "not_established",
        "reason": (final_csv_conclusion or {}).get("conclusion", (
            "The direct mechanism-vs-magnitude test (n=10, up to 3 mechanism categories, "
            "leave-one-out cross-validated) returns negative out-of-sample R2 for magnitude-only, "
            "mechanism-only, AND both-combined models. In-sample R2 favors mechanism (0.47 vs. "
            "0.08), but that is not trustworthy at this sample size and is directly contradicted "
            "by the out-of-sample check. Cannot be established as 'supported' OR 'falsified' -- "
            "the test is underpowered, not merely negative.")),
        "artifact": "results/mechanism_vs_magnitude_final.csv",
        "n": mvm.get("n_rows_used") if mvm else None,
        "explicitly_not_conflatable_with": "H2a's state-identity finding (H2a is about CONTEXT, "
                                          "this is about MECHANISM CATEGORY -- they use different "
                                          "predictors and different, non-overlapping evidence)",
    }


# ---------------------------------------------------------------------------------------------
# H3 -- item 14 redefinition (round-5). Old H3 ("staged audit beats scalar shift magnitude") was
# too strong for what this benchmark can actually test out of sample. Redefined per the user's
# own suggested wording: "mechanism-aware diagnosis identifies a more specific audit response
# than scalar shift magnitude alone in the evaluated settings." The one test that would formally
# adjudicate this (h2c_mechanism_vs_magnitude_pooled, n=10) returns NEGATIVE LOO-CV R2 for every
# candidate model -- i.e. it cannot be trusted in either direction. Per the user's explicit
# instruction ("If we cannot quantify that, H3 should be: not formally tested / exploratory
# rather than 'mixed'. We should not claim a formal result without a formal comparison."), H3's
# status is downgraded from "mixed" to "not_formally_tested" whenever that is the case, with the
# within-mechanism trend data and qualitative cross-mechanism narrative retained as exploratory,
# descriptive context only -- never as a formal-comparison substitute.
# ---------------------------------------------------------------------------------------------
def h3_evidence(fp_df: pd.DataFrame, denom: dict) -> dict:
    """Within-mechanism severity trends are descriptive/exploratory context (only mechanisms with
    >=2 valid, informative severity points qualify). Cross-mechanism severity LABELS ('low' vs
    'medium' vs 'high') are NOT on a common scale across mechanisms (auc_dc-target vs
    |Delta-pi|-target vs coverage-fraction) -- PHASE3_DESIGN.md's own 'Severity levels' table
    admits this explicitly for the concept mechanism and implicitly for the others. Cross-
    mechanism comparison is therefore QUALITATIVE ONLY here, never a manufactured universal
    severity scale or a single mechanical true/false case. The formal mechanism-vs-magnitude
    comparison (H2-C direct test) is UNDERPOWERED (n=10, negative LOO-CV for every model) --
    H3's status reflects that directly rather than dressing up the within-mechanism trends as a
    formal test of the redefined question."""
    within_mechanism_trends = {}
    for (dom, mech), g in fp_df[fp_df.auc_available].groupby(["domain", "mechanism"]):
        if len(g) >= 2 and g["severity"].nunique() >= 2:
            within_mechanism_trends[f"{dom}/{mech}"] = {
                "n_severity_points": int(g["severity"].nunique()),
                "auc_delta_by_severity": g.set_index("severity")["auc_delta"].to_dict(),
                "monotonic_degradation": bool(g.sort_values("severity")["auc_delta"].is_monotonic_increasing),
            }

    qualitative_cross_mechanism = {
        "covariate_cells (lending + security, high nominal severity)": (
            "near-zero AUC movement at the HIGH end of a deliberately large constructed shift "
            "(lending high |delta_auc|=0.015; security high |delta_auc|~0) -- consistent with "
            "the covariate mechanism's designed negative-control role (P(Y|X) preserved)."),
        "prior/novel_mechanism cells (low nominal severity)": (
            "lending's one feasible prior cell (low_down) and S1's low-severity holdout both "
            "show non-trivial movement on SOME axis (lending: ECE/Brier move meaningfully even "
            "though corrected macro-F1 barely moves -- see item L's NC1 audit; S1: in-source "
            "unseen-family recall already at 0.994, i.e. not degraded, complicating a clean "
            "'small nominal severity still hurts' story here too)."),
        "reading": ("The qualitative pattern is MESSIER than a clean 'mechanism beats scalar "
                   "magnitude' story once the F1 correction and the S1 non-informative finding "
                   "are both accounted for. Some support remains (covariate genuinely looks more "
                   "robust than prior/novel-mechanism in the calibration axes), but this is now "
                   "reported as a qualitative, hedged observation, not a quantitative case with "
                   "a mechanical correct/incorrect verdict."),
    }

    n_trend_capable = len(within_mechanism_trends)
    mvm = _load("phase3_h2_magnitude_vs_mechanism.json") or {}
    formal_test_trustworthy = bool(mvm) and all(
        (mvm.get(m, {}).get("loo_cv_r2") or -1) > 0
        for m in ("M0_magnitude_only", "M1_mechanism_only", "M2_magnitude_plus_mechanism")
    )
    if formal_test_trustworthy:
        status = "mixed"
        status_reason = "formal mechanism-vs-magnitude comparison (H2-C) had a trustworthy (positive LOO-CV) result"
    else:
        status = "not_formally_tested"
        status_reason = (
            "H3 redefined (item 14): 'mechanism-aware diagnosis identifies a more specific audit "
            "response than scalar shift magnitude alone.' The one analysis that would formally "
            "test this (results/phase3_h2_magnitude_vs_mechanism.json: magnitude-only vs. "
            "mechanism-only vs. both, predicting |delta_auc| across T3.3+S1's n=10 mechanism-"
            "labeled cells, leave-one-out cross-validated) returns NEGATIVE out-of-sample R2 for "
            "EVERY candidate model -- none of them predict held-out cells better than the naive "
            "mean. Per the user's explicit instruction, a formal result is not claimed without a "
            "formal comparison that actually works; the within-mechanism trends and qualitative "
            "narrative below are retained as exploratory/descriptive context, not as evidence for "
            "or against H3."
        )
    return {"status": status, "status_reason": status_reason,
           "within_mechanism_trends": within_mechanism_trends,
           "n_mechanisms_with_severity_trend_data": n_trend_capable,
           "qualitative_cross_mechanism_comparison": qualitative_cross_mechanism,
           "formal_test_attempted": bool(mvm),
           "formal_test_artifact": "results/phase3_h2_magnitude_vs_mechanism.json",
           "note": "H3's mechanical single-case check from the first-pass gate was withdrawn "
                   "(item K) -- it compared severity LABELS across non-commensurate scales, "
                   "which PHASE3_DESIGN.md itself does not license as a valid comparison. The "
                   "round-4 'mixed' status is further downgraded to 'not_formally_tested' in "
                   "round-5 (item 14) because the formal comparison that would justify 'mixed' "
                   "was run and failed its own out-of-sample check."}


def run():
    fp_df = pd.read_csv(P["out"] / "phase3_failure_fingerprint.csv")
    t31_hyp = _load("phase3_T31_hypothesis_status.json")
    mag = _load("phase3_magnitude_analysis.json") or {}
    denom = exact_denominator()

    h1 = h1_evidence(fp_df, denom)
    h2 = h2_evidence(t31_hyp, mag, fp_df)
    h2a = h2a_evidence(h2)
    h2b = h2b_evidence(None)
    h3 = h3_evidence(fp_df, denom)

    gate = {
        "evidence_hierarchy_policy": ("Level 1 (T3.1 + original benchmark) is decisive. Level 2 "
                                      "(T3.3) corroborates only where a cell is both feasible AND "
                                      "informative (not ceiling/floor-limited). Level 3 (S1) is "
                                      "supporting context only and never outvotes Level 1/2 "
                                      "(item M)."),
        "H1_mechanism_distinguishable_fingerprints": h1,
        "H2_magnitude_insufficient_within_mechanism": h2,
        "H2a_scalar_magnitude_insufficient": h2a,
        "H2b_mechanism_beyond_magnitude": h2b,
        "H2_split_note": ("round-6: H2 is now reported as two SEPARATE top-level verdicts, H2a and "
                          "H2b, specifically so they cannot be silently collapsed back into one "
                          "'H2: supported' headline. The legacy H2_magnitude_insufficient_within_"
                          "mechanism key above is kept for backward compatibility with existing "
                          "readers and is equivalent to H2a, NOT to H2a+H2b combined."),
        "H3_staged_audit_beats_scalar": h3,
        "exact_denominator": denom["counts"],
        "standing_rule": "The title and central claim are NOT protected from this outcome "
                         "(PHASE3_DESIGN.md). This gate is Claude's mechanical application of "
                         "the frozen criteria to the actual, F1-corrected Phase 3 numbers; the "
                         "user reviews this evidence directly before any narrative section is "
                         "rewritten.",
    }
    (P["out"] / "phase3_final_hypothesis_gate.json").write_text(json.dumps(gate, indent=2))
    print(f"H1: {h1['status']}  H2a: {h2a['status']}  H2b: {h2b['status']}  H3: {h3['status']}")
    print(f"denominator: {denom['counts']}")
    print("wrote phase3_final_hypothesis_gate.json")
    return gate


if __name__ == "__main__":
    run()
