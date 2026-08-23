"""Builds results/f1_metric_migration.csv: old (mislabeled positive-class F1, stored under the
key `macro_f1`) vs. new (correct) `macro_f1` + `f1_positive`, for every (domain, model, seed,
split) in the base audit layer. Old values read from a pre-fix snapshot taken before
audit/engine.py was corrected (2026-08-23, round-4 hostile audit); new values read from the
regenerated audit_*.json files. Reused by synthesis/phase3_f1_migration.py for the Phase 3-
specific version of this same comparison.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402

OLD_DIR = P["out"] / "_pre_f1_fix_snapshot"
DOMAINS = ["clinical", "nlp", "lending", "security"]


def run() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        old = json.loads((OLD_DIR / f"audit_{domain}.json").read_text())
        new = json.loads((P["out"] / f"audit_{domain}.json").read_text())
        for model, seeds in new["models"].items():
            old_seeds = old["models"].get(model, {})
            for seed, sd in seeds.items():
                old_sd = old_seeds.get(seed, {})
                for split, s in sd["splits"].items():
                    old_s = old_sd.get("splits", {}).get(split, {})
                    old_val = old_s.get("macro_f1")  # was actually f1_positive under the old code
                    new_macro = s.get("macro_f1")
                    new_f1pos = s.get("f1_positive")
                    delta = (new_macro - old_val) if (old_val is not None and new_macro is not None) else None
                    rows.append({
                        "domain": domain, "model": model, "seed": seed, "split": split,
                        "old_f1_field_value": old_val,
                        "old_field_was_actually": ("true_macro_f1 (NLP, multiclass -- unaffected)"
                                                    if domain == "nlp" else "f1_positive (mislabeled as macro_f1)"),
                        "new_macro_f1": new_macro, "new_f1_positive": new_f1pos,
                        "delta_macro_minus_old": delta,
                        "changed": bool(delta is not None and abs(delta) > 1e-9),
                    })
    df = pd.DataFrame(rows)
    df.to_csv(P["out"] / "f1_metric_migration.csv", index=False)
    print(f"wrote f1_metric_migration.csv ({len(df)} rows)")
    changed = df[df.changed]
    print(f"{len(changed)}/{len(df)} rows changed value (nlp rows should show changed=False -- multiclass unaffected)")
    if len(changed):
        print(changed[["domain", "model", "seed", "split", "old_f1_field_value", "new_macro_f1",
                       "delta_macro_minus_old"]].to_string())
    return df


if __name__ == "__main__":
    run()
