"""Standardized prediction schema shared by every domain adapter (PLAN.md §3).

One row per (example x subgroup_axis). Adapters emit results/predictions_{domain}.parquet
conforming to this schema; the audit engine consumes only conforming frames.
"""
import pandas as pd

from config import REQUIRED_COLUMNS

VALID_DOMAINS = {"clinical", "nlp", "lending", "security"}


def validate_predictions(df: pd.DataFrame) -> list[str]:
    """Return a list of violation messages; empty list means the frame conforms."""
    errors: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return errors  # column checks below would all crash

    if df.empty:
        errors.append("frame is empty")
        return errors

    bad_domain = set(df["domain"].unique()) - VALID_DOMAINS
    if bad_domain:
        errors.append(f"unknown domain values: {sorted(bad_domain)}")

    if not df["y_true"].isin([0, 1]).all():
        errors.append("y_true contains values outside {0, 1}")

    p = df["p_hat"]
    if p.isna().any() or (p < 0).any() or (p > 1).any():
        errors.append("p_hat outside [0, 1] or NaN")

    ok_split = df["split"].astype(str).str.match(r"^(source_test|target_.+)$")
    if not ok_split.all():
        bad = df.loc[~ok_split, "split"].unique()[:5]
        errors.append(f"split values must be 'source_test' or 'target_<name>': {list(bad)}")

    # NaN anywhere except class_label (nullable for binary domains)
    nan_cols = [c for c in REQUIRED_COLUMNS if c != "class_label" and df[c].isna().any()]
    if nan_cols:
        errors.append(f"NaNs in columns: {nan_cols}")

    # every subgroup_axis must carry >= 2 distinct subgroups (else no gap is defined)
    n_groups = df.groupby("subgroup_axis")["subgroup"].nunique()
    single = n_groups[n_groups < 2].index.tolist()
    if single:
        errors.append(f"subgroup_axis with <2 subgroups: {single}")

    # row_id unique within one logical prediction set
    key = ["domain", "model", "seed", "split", "subgroup_axis", "class_label"]
    dup = df.duplicated(subset=key + ["row_id"], keep=False)
    if dup.any():
        errors.append(f"{int(dup.sum())} duplicate row_id rows within {key}")

    return errors


def save_validated(df: pd.DataFrame, path) -> None:
    """Validate then write parquet; raise on any violation."""
    errs = validate_predictions(df)
    if errs:
        raise ValueError("schema violations: " + "; ".join(errs))
    df.to_parquet(path, index=False)
    print(f"wrote {len(df):,} rows -> {path}")
