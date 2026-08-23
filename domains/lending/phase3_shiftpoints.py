"""Phase 3A / T3.1 (docs/PHASE3_DESIGN.md ## T3.1): one frozen lending source model per seed,
scored on every valid (state x year) target cell. NO retraining per state -- each seed's model
is fit once (identical to domains/lending/train.py's lightgbm_temporal branch) and then reused,
unchanged, to score every included cell.

Inclusion gate is applied ONCE per (state, year) -- seed-independent, since it only depends on
how many raw panel rows exist in that slice -- and logged to lending_excluded_cells.json BEFORE
any metric is computed on included cells, exactly as frozen.

Delta-sign convention matches synthesis/tables.py::t2_master exactly (positive delta = target
is WORSE than source on that axis, for every metric):
    delta_auc      = auc_src - auc_tgt        (AUC: higher is better)
    delta_macro_f1 = f1_src  - f1_tgt         (F1: higher is better)
    delta_ece      = ece_tgt - ece_src        (ECE: lower is better)
    delta_brier    = brier_tgt - brier_src    (Brier: lower is better)
    delta_gap_auc  = gap_tgt  - gap_src       (subgroup gap: smaller is better)
delta_pi = pi_tgt - pi_src (signed; no existing engine.py precedent for this one -- documented
here since it is new to this script).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import P, SEEDS_NEW  # noqa: E402
from domains.lending.train import (  # noqa: E402
    FEATURES, _load_panel, _strat_sample, _split_holdout, _fit, CAP_TRAIN, CAP_TEST,
)
from audit.engine import _auc_se, _subgroup_gap, ECE_BINS  # noqa: E402
from audit.diagnosis import _domain_classifier_auc  # noqa: E402
from fairscope.core.calibration import expected_calibration_error  # noqa: E402
from fairscope.core.metrics import brier_score_loss  # noqa: E402

MIN_TOTAL = 500
MIN_SUBGROUP = 30
MIN_CLASS = 10
TARGET_YEARS = [2022, 2023, 2024]


def _gate(cell: pd.DataFrame) -> list[str]:
    """Exact three-part T3.1 inclusion gate. Returns failed-threshold reasons; empty = included."""
    reasons = []
    n = len(cell)
    if n < MIN_TOTAL:
        reasons.append(f"total_rows={n}<{MIN_TOTAL}")
    for label, sub in (("black", cell[cell.black == 1]), ("white", cell[cell.black == 0])):
        if len(sub) < MIN_SUBGROUP:
            reasons.append(f"{label}_rows={len(sub)}<{MIN_SUBGROUP}")
        n_pos, n_neg = int((sub.approved_clean == 1).sum()), int((sub.approved_clean == 0).sum())
        if n_pos < MIN_CLASS:
            reasons.append(f"{label}_positive={n_pos}<{MIN_CLASS}")
        if n_neg < MIN_CLASS:
            reasons.append(f"{label}_negative={n_neg}<{MIN_CLASS}")
    n_pos, n_neg = int((cell.approved_clean == 1).sum()), int((cell.approved_clean == 0).sum())
    if n_pos < MIN_CLASS:
        reasons.append(f"overall_positive={n_pos}<{MIN_CLASS}")
    if n_neg < MIN_CLASS:
        reasons.append(f"overall_negative={n_neg}<{MIN_CLASS}")
    return reasons


def _score(df: pd.DataFrame, model) -> dict:
    """AUC(+DeLong SE), macro_f1, f1_positive, ECE, Brier, race_black AUC gap -- same functions/
    convention audit/engine.py uses for every other domain's per-split aggregate.

    2026-08-23 correction (round-4 hostile audit): macro_f1 now uses average="macro" (the
    unweighted mean of BOTH classes' F1), matching the audit/engine.py fix. f1_positive (the
    positive/"approved" class alone, sklearn's average="binary" default) is reported alongside
    it -- this is what the pre-fix code silently called `macro_f1`."""
    p = np.clip(model.predict_proba(df[FEATURES])[:, 1], 0, 1)
    y = df["approved_clean"].astype(int).values
    auc, se = _auc_se(y, p)
    pred = (p >= 0.5).astype(int)
    f1_positive = f1_score(y, pred, zero_division=0)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    ece = expected_calibration_error(y, p, n_bins=ECE_BINS)
    brier = brier_score_loss(y, p)
    axdf = pd.DataFrame({"y_true": y, "p_hat": p,
                         "subgroup": np.where(df["black"].values == 1, "black", "white")})
    gap_auc, _ = _subgroup_gap(axdf, "auc")
    return dict(auc=auc, auc_se=se, macro_f1=macro_f1, f1_positive=f1_positive, ece=ece,
                brier=brier, gap_auc=gap_auc, pi=float(y.mean()), n=int(len(df)))


def build_grid_and_gate(panel: pd.DataFrame):
    states = sorted(int(s) for s in panel.state_fips.dropna().unique())
    included, excluded = [], []
    for s in states:
        for yr in TARGET_YEARS:
            cell = panel[(panel.state_fips == s) & (panel.year == yr)]
            reasons = _gate(cell)
            if reasons:
                excluded.append({"state_fips": s, "year": yr, "n_rows": int(len(cell)), "reasons": reasons})
            else:
                included.append((s, yr))
    return states, included, excluded


def run():
    panel = _load_panel()
    states, included, excluded = build_grid_and_gate(panel)
    (P["out"] / "lending_excluded_cells.json").write_text(json.dumps({
        "gate": {"min_total_rows": MIN_TOTAL, "min_subgroup_rows": MIN_SUBGROUP,
                 "min_class_count": MIN_CLASS,
                 "note": "income_quartile is not a gating axis -- race_black is the sole "
                         "gating subgroup axis (PHASE3_DESIGN.md T3.1)."},
        "n_states": len(states), "n_years": len(TARGET_YEARS),
        "n_cells_total": len(states) * len(TARGET_YEARS),
        "n_included": len(included), "n_excluded": len(excluded),
        "excluded": excluded,
    }, indent=2))
    print(f"grid: {len(states)} states x {len(TARGET_YEARS)} years = "
          f"{len(states) * len(TARGET_YEARS)} cells; included={len(included)} excluded={len(excluded)}")

    rows = []
    for seed in SEEDS_NEW:
        src = panel[panel.year.isin([2020, 2021])]
        src_tr, src_te = _split_holdout(src, seed)
        train = _strat_sample(src_tr, CAP_TRAIN, seed)
        model = _fit(train, seed)
        src_te_c = _strat_sample(src_te, CAP_TEST, seed)
        src_score = _score(src_te_c, model)
        Xs = src_te_c[FEATURES].to_numpy()
        print(f"seed {seed}: source_test AUC={src_score['auc']:.4f} n={src_score['n']:,} -- scoring {len(included)} cells")

        for i, (s, yr) in enumerate(included):
            cell = panel[(panel.state_fips == s) & (panel.year == yr)]
            cell_score = _score(cell, model)
            Xt = cell[FEATURES].to_numpy()
            auc_dc = _domain_classifier_auc(Xs, Xt, seed=seed)
            rows.append({
                "seed": seed, "state_fips": s, "year": yr, "n_rows": cell_score["n"],
                "auc_dc": auc_dc,
                "pi_src": src_score["pi"], "pi_tgt": cell_score["pi"],
                "delta_pi": cell_score["pi"] - src_score["pi"],
                "auc_src": src_score["auc"], "auc_tgt": cell_score["auc"],
                "delta_auc": src_score["auc"] - cell_score["auc"],
                "macro_f1_src": src_score["macro_f1"], "macro_f1_tgt": cell_score["macro_f1"],
                "delta_macro_f1": src_score["macro_f1"] - cell_score["macro_f1"],
                "f1_positive_src": src_score["f1_positive"], "f1_positive_tgt": cell_score["f1_positive"],
                "delta_f1_positive": src_score["f1_positive"] - cell_score["f1_positive"],
                "ece_src": src_score["ece"], "ece_tgt": cell_score["ece"],
                "delta_ece": cell_score["ece"] - src_score["ece"],
                "brier_src": src_score["brier"], "brier_tgt": cell_score["brier"],
                "delta_brier": cell_score["brier"] - src_score["brier"],
                "gap_auc_src": src_score["gap_auc"], "gap_auc_tgt": cell_score["gap_auc"],
                "delta_gap_auc": cell_score["gap_auc"] - src_score["gap_auc"],
            })
            if (i + 1) % 10 == 0:
                print(f"  seed {seed}: {i + 1}/{len(included)} cells scored")

    out = pd.DataFrame(rows)
    out.to_csv(P["out"] / "lending_shiftpoints.csv", index=False)
    print(f"wrote {len(out)} rows -> results/lending_shiftpoints.csv")
    return out


def rescore_fast(existing_csv=None) -> pd.DataFrame:
    """Fast-path rescore used ONLY for the 2026-08-23 F1-definition correction: auc_dc is
    computed by _domain_classifier_auc() from feature matrices alone and never touches the
    outcome model's predictions or F1 at all, so it is provably unaffected by the macro_f1 fix
    and does not need to be recomputed (this is what makes the 149-cell x 3-seed domain-
    classifier refit, the expensive ~20-30 minute part of run(), safely skippable here). Reuses
    each row's existing auc_dc; recomputes every score()-derived column (auc, f1_positive,
    macro_f1, ece, brier, gap_auc, pi) fresh from a refit of the same frozen per-seed model.
    NOT the canonical entry point -- run() remains that; this exists so a metric-only fix does
    not force an unnecessary multi-minute-to-tens-of-minutes AUC_dc recomputation."""
    existing_csv = existing_csv or (P["out"] / "lending_shiftpoints.csv")
    old = pd.read_csv(existing_csv)
    auc_dc_by_row = {(r.seed, r.state_fips, r.year): r.auc_dc for r in old.itertuples()}

    panel = _load_panel()
    included = sorted({(int(r.state_fips), int(r.year)) for r in old.itertuples()})
    rows = []
    for seed in SEEDS_NEW:
        src = panel[panel.year.isin([2020, 2021])]
        src_tr, src_te = _split_holdout(src, seed)
        train = _strat_sample(src_tr, CAP_TRAIN, seed)
        model = _fit(train, seed)
        src_te_c = _strat_sample(src_te, CAP_TEST, seed)
        src_score = _score(src_te_c, model)
        print(f"seed {seed}: source_test AUC={src_score['auc']:.4f} macro_f1={src_score['macro_f1']:.4f} "
              f"(was f1_positive={src_score['f1_positive']:.4f} under the old, mislabeled convention)")

        for s, yr in included:
            key = (seed, s, yr)
            if key not in auc_dc_by_row:
                continue  # cell wasn't in the prior run (shouldn't happen; grid/gate unchanged)
            cell = panel[(panel.state_fips == s) & (panel.year == yr)]
            cell_score = _score(cell, model)
            rows.append({
                "seed": seed, "state_fips": s, "year": yr, "n_rows": cell_score["n"],
                "auc_dc": auc_dc_by_row[key],
                "pi_src": src_score["pi"], "pi_tgt": cell_score["pi"],
                "delta_pi": cell_score["pi"] - src_score["pi"],
                "auc_src": src_score["auc"], "auc_tgt": cell_score["auc"],
                "delta_auc": src_score["auc"] - cell_score["auc"],
                "macro_f1_src": src_score["macro_f1"], "macro_f1_tgt": cell_score["macro_f1"],
                "delta_macro_f1": src_score["macro_f1"] - cell_score["macro_f1"],
                "f1_positive_src": src_score["f1_positive"], "f1_positive_tgt": cell_score["f1_positive"],
                "delta_f1_positive": src_score["f1_positive"] - cell_score["f1_positive"],
                "ece_src": src_score["ece"], "ece_tgt": cell_score["ece"],
                "delta_ece": cell_score["ece"] - src_score["ece"],
                "brier_src": src_score["brier"], "brier_tgt": cell_score["brier"],
                "delta_brier": cell_score["brier"] - src_score["brier"],
                "gap_auc_src": src_score["gap_auc"], "gap_auc_tgt": cell_score["gap_auc"],
                "delta_gap_auc": cell_score["gap_auc"] - src_score["gap_auc"],
            })

    out = pd.DataFrame(rows)
    assert len(out) == len(old), (len(out), len(old))
    out.to_csv(P["out"] / "lending_shiftpoints.csv", index=False)
    print(f"rescored {len(out)} rows (auc_dc reused, all else recomputed) -> results/lending_shiftpoints.csv")
    return out


if __name__ == "__main__":
    run()
