"""T3.3 lending cells (docs/PHASE3_DESIGN.md ## T3.3 'Formal per-cell specification'): covariate
(negative-control role), prior/label, and concept (natural-subpopulation, discovered not dialed).
Novel-class is N/A for lending by construction (binary approve/deny has no unseen-class analogue)
and is recorded as such, not forced.

Uses seed=42 only (the paper's primary reporting seed throughout, e.g. audit/engine.py's
ci_seed=42) -- T3.3 is explicitly the "compact" study (PHASE3_DESIGN.md section title), not a
full multi-seed replication like T3.1. Reuses the exact frozen lightgbm_temporal seed-42 model
(same _fit call as domains/lending/train.py and domains/lending/phase3_shiftpoints.py) and
phase3_shiftpoints._score for all ΔAUC/Δmacro-F1/ΔECE/ΔBrier/ΔG reporting, against that seed's own
source_test, exactly matching every other lending number in this benchmark.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import P  # noqa: E402
from domains.lending.train import FEATURES, _load_panel, _strat_sample, _split_holdout, _fit, CAP_TRAIN, CAP_TEST  # noqa: E402
from domains.lending.phase3_shiftpoints import _score  # noqa: E402
from synthesis.phase3_mechanism_lib import calibrate_covariate_resample, calibrate_prior_resample  # noqa: E402

SEED = 42
COVARIATE_COLS = ["ltv", "purpose_refi", "aus_automated"]
PRIOR_STRATUM_YEAR = 2023


def _fit_seed42_model(panel: pd.DataFrame):
    src = panel[panel.year.isin([2020, 2021])]
    src_tr, src_te = _split_holdout(src, SEED)
    train = _strat_sample(src_tr, CAP_TRAIN, SEED)
    model = _fit(train, SEED)
    src_te_c = _strat_sample(src_te, CAP_TEST, SEED)
    return model, src_te_c


def _deltas(src_score: dict, tgt_score: dict) -> dict:
    return {
        "auc_src": src_score["auc"], "auc_tgt": tgt_score["auc"],
        "delta_auc": src_score["auc"] - tgt_score["auc"],
        "macro_f1_src": src_score["macro_f1"], "macro_f1_tgt": tgt_score["macro_f1"],
        "delta_macro_f1": src_score["macro_f1"] - tgt_score["macro_f1"],
        "f1_positive_src": src_score["f1_positive"], "f1_positive_tgt": tgt_score["f1_positive"],
        "delta_f1_positive": src_score["f1_positive"] - tgt_score["f1_positive"],
        "ece_src": src_score["ece"], "ece_tgt": tgt_score["ece"],
        "delta_ece": tgt_score["ece"] - src_score["ece"],
        "brier_src": src_score["brier"], "brier_tgt": tgt_score["brier"],
        "delta_brier": tgt_score["brier"] - src_score["brier"],
        "gap_auc_src": src_score["gap_auc"], "gap_auc_tgt": tgt_score["gap_auc"],
        "delta_gap_auc": tgt_score["gap_auc"] - src_score["gap_auc"],
    }


def cell_covariate(panel, model, src_score, pool_year=2023) -> dict:
    pool = _strat_sample(panel[panel.year == pool_year], CAP_TEST, SEED)
    calib = calibrate_covariate_resample(pool, COVARIATE_COLS, FEATURES, SEED)
    out = {"mechanism": "covariate", "status": "OK", "pool_year": pool_year, "propensity_cols": COVARIATE_COLS,
          "severities": {}}
    for level, c in calib.items():
        tgt_score = _score(c["resampled"], model)
        out["severities"][level] = {
            "target_auc_dc": c["target_auc_dc"], "achieved_auc_dc": c["achieved_auc_dc"],
            "temperature": c["temperature"], "feasible": c["feasible"], "n": c["n"],
            **_deltas(src_score, tgt_score),
        }
    return out


def cell_prior(panel, model, src_score, stratum_year=PRIOR_STRATUM_YEAR) -> dict:
    yr_df = _strat_sample(panel[panel.year == stratum_year], CAP_TEST, SEED)
    stratum = yr_df[(yr_df["purpose_purchase"] == 1)].reset_index(drop=True)
    from domains.lending.train import _income_quartile
    inc_edges = np.quantile(yr_df["income"], [0, .25, .5, .75, 1.0])
    q = _income_quartile(stratum["income"], inc_edges)
    stratum = stratum[q == "q2"].reset_index(drop=True)
    calib = calibrate_prior_resample(stratum, "approved_clean", SEED)
    out = {"mechanism": "prior", "status": "OK", "stratum": f"year={stratum_year}, purpose_purchase=1, income_quartile=q2",
          "n_stratum": int(len(stratum)), "natural_pi": calib["natural_pi"], "severities": {}}
    for level in ("low", "medium", "high"):
        for direction in ("up", "down"):
            key = f"{level}_{direction}"
            c = calib[key]
            tgt_score = _score(c["resampled"], model)
            out["severities"][key] = {
                "target_delta_pi": c["target_delta_pi"], "achieved_delta_pi": c["achieved_delta_pi"],
                "direction": direction, "feasible": c["feasible"], "reaches_target": c["reaches_target"],
                "replacement_frac": c["replacement_frac"], "n": c["n"],
                **_deltas(src_score, tgt_score),
            }
    return out


def cell_concept_natural() -> dict:
    """Natural-subpopulation concept-shift cell: the T3.1 (state, year) shiftpoint whose observed
    delta_ece deviates most from what the fitted OLS(delta_ece ~ auc_dc) line predicts -- the
    biggest outlier from the covariate-magnitude story, seed=42 only, discovered not dialed."""
    sp_fp = P["out"] / "lending_shiftpoints.csv"
    if not sp_fp.exists():
        return {"mechanism": "concept_natural", "status": "PENDING",
               "note": "requires results/lending_shiftpoints.csv (T3.1) to exist first"}
    df = pd.read_csv(sp_fp)
    d42 = df[df.seed == SEED].reset_index(drop=True)
    x = d42["auc_dc"].to_numpy(); y = d42["delta_ece"].to_numpy()
    b1, b0 = np.polyfit(x, y, 1)
    resid = y - (b1 * x + b0)
    i = int(np.argmax(np.abs(resid)))
    row = d42.iloc[i]
    return {
        "mechanism": "concept_natural", "status": "FOUND",
        "construction": "T3.1 (state,year) shiftpoint, seed=42, with the largest |residual| from "
                        "the fitted OLS(delta_ece ~ auc_dc) line -- nothing synthetic; the model "
                        "and data are both real and unmodified.",
        "state_fips": int(row["state_fips"]), "year": int(row["year"]),
        "auc_dc": float(row["auc_dc"]), "ols_predicted_delta_ece": float(b1 * row["auc_dc"] + b0),
        "actual_delta_ece": float(row["delta_ece"]), "residual_magnitude": float(np.abs(resid[i])),
        "delta_auc": float(row["delta_auc"]), "delta_macro_f1": float(row["delta_macro_f1"]),
        "delta_brier": float(row["delta_brier"]), "delta_gap_auc": float(row["delta_gap_auc"]),
        "measured_severity_note": "severity is DISCOVERED (this residual magnitude), not dialed -- "
                                  "PHASE3_DESIGN.md T3.3 explicitly forbids forcing low/med/high "
                                  "levels for this mechanism.",
    }


def run():
    panel = _load_panel()
    model, src_te_c = _fit_seed42_model(panel)
    src_score = _score(src_te_c, model)
    print(f"seed42 lending source_test: AUC={src_score['auc']:.4f}")

    cov = cell_covariate(panel, model, src_score)
    print(f"lending-covariate: achieved auc_dc per severity = "
          f"{[(k, round(v['achieved_auc_dc'],3), v['feasible']) for k,v in cov['severities'].items()]}")

    pri = cell_prior(panel, model, src_score)
    print(f"lending-prior: n_stratum={pri['n_stratum']} natural_pi={pri['natural_pi']:.3f}")

    con = cell_concept_natural()
    print(f"lending-concept-natural: {con.get('status')}")

    novel = {"mechanism": "novel_class", "status": "N_A",
            "reason": "Binary approve/deny has no natural analogue of an unseen class; forcing "
                      "one would require inventing a category that does not exist in mortgage "
                      "underwriting decisions (PHASE3_DESIGN.md T3.3 validity table)."}

    out = {"domain": "lending", "seed": SEED, "cells": {"covariate": cov, "prior": pri,
                                                        "concept_natural": con, "novel_class": novel}}
    (P["out"] / "phase3_T33_lending.json").write_text(json.dumps(out, indent=2))
    print("wrote results/phase3_T33_lending.json")
    return out


if __name__ == "__main__":
    run()
