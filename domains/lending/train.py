"""Lending domain (NEW analysis): HMDA approval prediction under temporal + geographic shift.

Trains here (unlike clinical/nlp which reuse saved predictions). Data: the 42.3M-row
features_panel.parquet from the CATE-HMDA repo (PLAN.md §2c). Nothing on predictive shift
was published for HMDA, so this is genuinely new.

Panel facts verified 2026-07-12:
  - `black`: 1=Black (applicant_race_1==3), 0=White non-Hispanic (==5). Panel pre-filtered
    to these two groups only (42.30M rows). Raw approval 83.2% White vs 68.3% Black = 14.9pp,
    reproducing P2's 14.95pp raw gap.
  - outcome `approved_clean` (1=approved). Features = X_BASE_NOGEO minus {year, post_tightening}
    so no race, no geography, and no temporal-index feature leaks the shift.

Two experiments, emitted as two models (each with its own source_test):
  - lightgbm_temporal: train 2020-2021 -> source_test(heldout 20-21), target_2022/2023/2024
  - lightgbm_geo:       train 60% of states -> source_test(heldout rows), target_heldout_states
Subgroup axes: race_black (black|white), income_quartile (q1..q4, cut on the training split).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import P, SEEDS_NEW  # noqa: E402
from schema import save_validated  # noqa: E402

CAP_TRAIN = 2_000_000   # stratified train cap (memory guard; P1 pipeline used similar)
CAP_TEST = 200_000      # per test cell
RNG = np.random.default_rng(0)

_fs = json.loads(Path(P["hmda_feature_sets"]).read_text())
FEATURES = [f for f in _fs["X_BASE_NOGEO"] if f not in ("year", "post_tightening")]
LOAD_COLS = FEATURES + ["black", "approved_clean", "year", "state_fips", "income"]


def _load_panel() -> pd.DataFrame:
    df = pd.read_parquet(P["hmda_features"], columns=list(dict.fromkeys(LOAD_COLS)))
    for c in df.columns:  # downcast for the 16 GB RAM guard
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "int64":
            df[c] = df[c].astype("int32")
    return df


def _strat_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    # stratify by black x approved_clean so rare cells survive
    g = df.groupby(["black", "approved_clean"], observed=True)
    frac = n / len(df)
    return g.sample(frac=frac, random_state=seed).reset_index(drop=True)


def _income_quartile(income: pd.Series, edges: np.ndarray) -> np.ndarray:
    q = np.digitize(income.values, edges[1:-1])  # 0..3
    return np.array(["q1", "q2", "q3", "q4"])[q]


def _fit(train: pd.DataFrame, seed: int) -> LGBMClassifier:
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                       subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                       random_state=seed, n_jobs=-1, verbose=-1)
    assert not set(FEATURES) & {"black", "state_fips", "year"}, "race/geo leaked into features"
    m.fit(train[FEATURES], train["approved_clean"])
    return m


def _emit(model_name, seed, split, test, model, inc_edges):
    p = model.predict_proba(test[FEATURES])[:, 1]
    y = test["approved_clean"].astype(int).values
    n = len(test)
    base = dict(domain="lending", model=model_name, seed=seed, split=split,
                y_true=y, class_label=None, p_hat=np.clip(p, 0, 1))
    race = pd.DataFrame({**base, "subgroup_axis": "race_black",
                         "subgroup": np.where(test["black"].values == 1, "black", "white"),
                         "row_id": [f"ld_{model_name}_{split}_s{seed}_r{i}" for i in range(n)]})
    inc = pd.DataFrame({**base, "subgroup_axis": "income_quartile",
                        "subgroup": _income_quartile(test["income"], inc_edges),
                        "row_id": [f"ld_{model_name}_{split}_s{seed}_q{i}" for i in range(n)]})
    return [race, inc]


def build() -> pd.DataFrame:
    panel = _load_panel()
    print(f"panel loaded: {len(panel):,} rows, {len(FEATURES)} features")
    frames = []

    for seed in SEEDS_NEW:
        # ---- Experiment A: temporal ----
        src = panel[panel.year.isin([2020, 2021])]
        src_tr, src_te = _split_holdout(src, seed)
        train = _strat_sample(src_tr, CAP_TRAIN, seed)
        inc_edges = np.quantile(train["income"], [0, .25, .5, .75, 1.0])
        model = _fit(train, seed)
        frames += _emit("lightgbm_temporal", seed, "source_test",
                        _strat_sample(src_te, CAP_TEST, seed), model, inc_edges)
        for yr in (2022, 2023, 2024):
            cell = _strat_sample(panel[panel.year == yr], CAP_TEST, seed)
            frames += _emit("lightgbm_temporal", seed, f"target_{yr}", cell, model, inc_edges)

        # ---- Experiment B: geographic ----
        states = np.sort(panel.state_fips.dropna().unique())
        rng = np.random.default_rng(seed)
        train_states = set(rng.choice(states, size=int(0.6 * len(states)), replace=False).tolist())
        in_tr = panel.state_fips.isin(train_states)
        gtr_all = panel[in_tr]
        gtr, gte = _split_holdout(gtr_all, seed)
        gtrain = _strat_sample(gtr, CAP_TRAIN, seed)
        ginc = np.quantile(gtrain["income"], [0, .25, .5, .75, 1.0])
        gmodel = _fit(gtrain, seed)
        frames += _emit("lightgbm_geo", seed, "source_test",
                        _strat_sample(gte, CAP_TEST, seed), gmodel, ginc)
        held = _strat_sample(panel[~in_tr], CAP_TEST, seed)
        frames += _emit("lightgbm_geo", seed, "target_heldout_states", held, gmodel, ginc)
        print(f"  seed {seed} done")

    return pd.concat(frames, ignore_index=True)


def _split_holdout(df: pd.DataFrame, seed: int, frac_test: float = 0.2):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(len(df) * (1 - frac_test))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def main():
    df = build()
    save_validated(df, P["out"] / "predictions_lending.parquet")
    from sklearn.metrics import roc_auc_score
    print("\n-- lending AUC + Black/White gap (seed 42, race axis) --")
    d = df[(df.seed == 42) & (df.subgroup_axis == "race_black")]
    for model in ["lightgbm_temporal", "lightgbm_geo"]:
        for split in sorted(d[d.model == model].split.unique()):
            s = d[(d.model == model) & (d.split == split)]
            auc = roc_auc_score(s.y_true, s.p_hat)
            gaps = {g: roc_auc_score(x.y_true, x.p_hat) for g, x in s.groupby("subgroup")
                    if x.y_true.nunique() == 2}
            gap = abs(gaps.get("black", np.nan) - gaps.get("white", np.nan))
            print(f"  {model:18s} {split:22s} AUC={auc:.4f}  B/W AUC gap={gap:.4f}")


if __name__ == "__main__":
    main()
