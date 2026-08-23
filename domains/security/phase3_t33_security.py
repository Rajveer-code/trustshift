"""T3.3 security cells (docs/PHASE3_DESIGN.md ## T3.3 'Formal per-cell specification'): covariate
(within reflection_volumetric only, negative-control role), prior/label (within a single fixed
mechanism, syn-only), concept (targeted feasibility check run FIRST, per the design doc's own
"requires a targeted check of file/tool metadata before deciding" -- not assumed N/A from the
design doc's prose without re-verifying against the actual current data), and novel-mechanism
(reused directly from S1, no new construction).

Uses seed=42 only (compact study, matches domains/lending/phase3_t33_lending.py's scope choice
and the paper's primary reporting seed elsewhere). Reuses run_pipeline.py's load/harmonize
primitives and provenance_audit.py's MECHANISM_CATEGORY -- no retraining logic duplicated from
phase3_mechanism_holdout.py; a small local fit+eval helper is used here because T3.3's covariate/
prior cells do not need the seen/unseen-family split S1's _fit_and_eval computes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import P, SEED, ECE_BINS  # noqa: E402
from domains.security.run_pipeline import (  # noqa: E402
    _load_parquet_dir, harmonize, to_binary, _family, _cap, SAMPLE_N_PER_CLASS,
)
from domains.security.provenance_audit import MECHANISM_CATEGORY  # noqa: E402
from fairscope.core.calibration import expected_calibration_error  # noqa: E402
from fairscope.core.metrics import brier_score_loss  # noqa: E402
from synthesis.phase3_mechanism_lib import calibrate_covariate_resample, calibrate_prior_resample  # noqa: E402

DURATION_BYTE_KEYWORDS = ("duration", "byte")


def load():
    import truststore
    truststore.inject_into_ssl()
    import kagglehub
    p19 = kagglehub.dataset_download("dhoogla/cicddos2019")
    df19, lab19 = harmonize(_load_parquet_dir(p19))
    df19["_ybin"] = to_binary(df19[lab19]); df19["_fam"] = _family(df19[lab19])
    FEATURES = sorted(df19.drop(columns=[lab19]).select_dtypes("number").columns)
    is_train_file = df19["_srcfile"].str.endswith("-training")
    is_test_file = df19["_srcfile"].str.endswith("-testing")
    df19_train_raw = df19[is_train_file].reset_index(drop=True)
    df19_test_raw = df19[is_test_file].reset_index(drop=True)
    return df19_train_raw, df19_test_raw, FEATURES


def _fit_eval(train_df, test_df, FEATURES, seed):
    train_c = _cap(train_df, "_fam", SAMPLE_N_PER_CLASS, seed)
    med = train_c[FEATURES].median()
    Xtr = train_c[FEATURES].fillna(med).to_numpy(np.float32)
    ytr = train_c["_ybin"].to_numpy()
    scaler = MinMaxScaler().fit(Xtr)
    clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8,
                        colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                        random_state=seed, n_jobs=-1, verbosity=0)
    clf.fit(scaler.transform(Xtr), ytr, sample_weight=compute_sample_weight("balanced", ytr))

    def _score(d):
        X = np.clip(scaler.transform(d[FEATURES].fillna(med).to_numpy(np.float32)), 0, 1)
        y = d["_ybin"].to_numpy()
        p = np.clip(clf.predict_proba(X)[:, 1], 0, 1)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
        pred = (p >= 0.5).astype(int)
        f1_positive = f1_score(y, pred, zero_division=0)
        macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
        ece = expected_calibration_error(y, p, n_bins=ECE_BINS)
        brier = brier_score_loss(y, p)
        return {"n": int(len(d)), "auc": auc, "macro_f1": float(macro_f1),
               "f1_positive": float(f1_positive), "ece": float(ece), "brier": float(brier)}
    return clf, scaler, med, _score


def _deltas(src, tgt):
    return {"auc_src": src["auc"], "auc_tgt": tgt["auc"], "delta_auc": src["auc"] - tgt["auc"],
           "macro_f1_src": src["macro_f1"], "macro_f1_tgt": tgt["macro_f1"],
           "delta_macro_f1": src["macro_f1"] - tgt["macro_f1"],
           "f1_positive_src": src["f1_positive"], "f1_positive_tgt": tgt["f1_positive"],
           "delta_f1_positive": src["f1_positive"] - tgt["f1_positive"],
           "ece_src": src["ece"], "ece_tgt": tgt["ece"], "delta_ece": tgt["ece"] - src["ece"],
           "brier_src": src["brier"], "brier_tgt": tgt["brier"], "delta_brier": tgt["brier"] - src["brier"]}


def _find_duration_byte_cols(FEATURES: list[str]) -> list[str]:
    return [c for c in FEATURES if any(k in c.lower() for k in DURATION_BYTE_KEYWORDS)]


def cell_covariate(df19_train_raw, df19_test_raw, FEATURES, seed=SEED) -> dict:
    prop_cols = _find_duration_byte_cols(FEATURES)
    reflection_fams = [f for f, m in MECHANISM_CATEGORY.items() if m == "reflection_volumetric"]
    # binary task needs benign rows too -- "reflection_volumetric only" fixes which ATTACK
    # mechanism is present, it does not mean attack-only rows (that would leave a single-class
    # training set, as a first pass here caught: XGBoost raised "Invalid classes ... got [1]").
    keep_fams = reflection_fams + ["benign"]
    train_pool = df19_train_raw[df19_train_raw["_fam"].isin(keep_fams)].reset_index(drop=True)
    test_pool = df19_test_raw[df19_test_raw["_fam"].isin(keep_fams)].reset_index(drop=True)
    if not prop_cols or len(train_pool) < 500:
        return {"mechanism": "covariate", "status": "INFEASIBLE",
               "reason": f"prop_cols_found={prop_cols}, n_reflection_train_rows={len(train_pool)}"}
    clf, scaler, med, score_fn = _fit_eval(train_pool, test_pool, FEATURES, seed)
    src_score = score_fn(test_pool)
    calib = calibrate_covariate_resample(test_pool, prop_cols, FEATURES, seed)
    out = {"mechanism": "covariate", "status": "OK", "propensity_cols": prop_cols,
          "reflection_families": reflection_fams, "n_train": int(len(train_pool)), "severities": {}}
    for level, c in calib.items():
        tgt_score = score_fn(c["resampled"])
        out["severities"][level] = {
            "target_auc_dc": c["target_auc_dc"], "achieved_auc_dc": c["achieved_auc_dc"],
            "temperature": c["temperature"], "feasible": c["feasible"], "n": c["n"],
            **_deltas(src_score, tgt_score)}
    return out


def cell_prior(df19_train_raw, df19_test_raw, FEATURES, seed=SEED, mechanism="syn") -> dict:
    train_pool = df19_train_raw[df19_train_raw["_fam"].isin([mechanism, "benign"])].reset_index(drop=True)
    test_pool = df19_test_raw[df19_test_raw["_fam"].isin([mechanism, "benign"])].reset_index(drop=True)
    if len(train_pool) < 500:
        return {"mechanism": "prior", "status": "INFEASIBLE", "reason": f"n_train={len(train_pool)}"}
    clf, scaler, med, score_fn = _fit_eval(train_pool, test_pool, FEATURES, seed)
    src_score = score_fn(test_pool)
    calib = calibrate_prior_resample(test_pool, "_ybin", seed)
    out = {"mechanism": "prior", "status": "OK", "fixed_mechanism": mechanism,
          "n_train": int(len(train_pool)), "natural_pi": calib["natural_pi"], "severities": {}}
    for level in ("low", "medium", "high"):
        for direction in ("up", "down"):
            key = f"{level}_{direction}"
            c = calib[key]
            tgt_score = score_fn(c["resampled"])
            out["severities"][key] = {
                "target_delta_pi": c["target_delta_pi"], "achieved_delta_pi": c["achieved_delta_pi"],
                "direction": direction, "feasible": c["feasible"], "reaches_target": c["reaches_target"],
                "replacement_frac": c["replacement_frac"], "n": c["n"], **_deltas(src_score, tgt_score)}
    return out


def cell_concept_check(df19_train_raw: pd.DataFrame) -> dict:
    """Targeted feasibility check (PHASE3_DESIGN.md: 'requires a targeted check of file/tool
    metadata before deciding'): does any attack family have MORE THAN ONE distinct native
    training-file source? Two different capture files producing the SAME nominal family label
    would be the natural candidate for 'two variants of the same nominal attack generated by
    different tools/parameters' the design doc asks about. If every family maps to exactly one
    source file, no such natural variant exists in this dataset and the cell is N/A."""
    by_fam = df19_train_raw.groupby("_fam")["_srcfile"].nunique()
    multi_file_fams = {str(f): int(n) for f, n in by_fam.items() if n > 1 and f != "benign"}
    if multi_file_fams:
        return {"mechanism": "concept", "status": "VALID",
               "finding": f"families with >1 distinct native training-file source: {multi_file_fams}",
               "note": "candidate natural concept-shift construction exists; not built out further "
                       "in this pass given Phase 3 time/compute scope -- flagged for a follow-up run."}
    return {"mechanism": "concept", "status": "N_A",
           "finding": "every non-benign attack family in df19_train_raw maps to exactly one "
                      "native training-file source -- no two-variants-of-the-same-label "
                      "candidate exists in this dataset as loaded.",
           "reason": "PHASE3_DESIGN.md T3.3: 'if unavailable, mark N/A rather than synthetically "
                     "perturb features to fabricate a concept shift.'"}


def cell_novel_mechanism() -> dict:
    fp = P["out"] / "security_mechanism_ladder.csv"
    if not fp.exists():
        return {"mechanism": "novel_mechanism", "status": "PENDING",
               "note": "requires results/security_mechanism_ladder.csv (S1) to exist first"}
    return {"mechanism": "novel_mechanism", "status": "REUSED_FROM_S1",
           "note": "PHASE3_DESIGN.md T3.3: this cell reuses the S1 severity ladder directly "
                   "rather than building a redundant parallel experiment.",
           "artifact": "results/security_mechanism_ladder.csv"}


def run():
    df19_train_raw, df19_test_raw, FEATURES = load()
    cov = cell_covariate(df19_train_raw, df19_test_raw, FEATURES)
    print(f"security-covariate: {cov.get('status')}", cov.get("propensity_cols", cov.get("reason")))
    pri = cell_prior(df19_train_raw, df19_test_raw, FEATURES)
    print(f"security-prior: {pri.get('status')} natural_pi={pri.get('natural_pi')}")
    con = cell_concept_check(df19_train_raw)
    print(f"security-concept: {con['status']}")
    nov = cell_novel_mechanism()
    print(f"security-novel-mechanism: {nov['status']}")

    out = {"domain": "security", "seed": SEED,
          "cells": {"covariate": cov, "prior": pri, "concept": con, "novel_mechanism": nov}}
    (P["out"] / "phase3_T33_security.json").write_text(json.dumps(out, indent=2))
    print("wrote results/phase3_T33_security.json")
    return out


if __name__ == "__main__":
    run()
