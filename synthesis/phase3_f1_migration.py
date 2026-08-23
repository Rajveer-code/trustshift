"""Item E (Phase 3-specific): results/phase3_f1_metric_migration.csv, comparing T3.1's old
(mislabeled positive-class F1) delta against the corrected delta_macro_f1, per (state, year,
seed) cell. Old snapshot taken before domains/lending/phase3_shiftpoints.py was corrected.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402


def run() -> pd.DataFrame:
    old = pd.read_csv(P["out"] / "_pre_f1_fix_snapshot" / "lending_shiftpoints_old.csv")
    new = pd.read_csv(P["out"] / "lending_shiftpoints.csv")
    m = old[["seed", "state_fips", "year", "delta_macro_f1"]].merge(
        new[["seed", "state_fips", "year", "delta_macro_f1", "delta_f1_positive"]],
        on=["seed", "state_fips", "year"], suffixes=("_old_mislabeled", "_new_corrected"))
    m["old_field_was_actually"] = "f1_positive (mislabeled as macro_f1)"
    m["delta_of_deltas"] = m["delta_macro_f1_new_corrected"] - m["delta_macro_f1_old_mislabeled"]
    m["sign_flipped"] = (m["delta_macro_f1_old_mislabeled"] > 0) != (m["delta_macro_f1_new_corrected"] > 0)
    m.to_csv(P["out"] / "phase3_f1_metric_migration.csv", index=False)
    print(f"wrote phase3_f1_metric_migration.csv ({len(m)} rows)")
    print(f"sign flipped (degrades->improves or vice versa) in {m['sign_flipped'].sum()}/{len(m)} cells")
    print(m["delta_macro_f1_old_mislabeled"].describe())
    print(m["delta_macro_f1_new_corrected"].describe())
    return m


if __name__ == "__main__":
    run()
