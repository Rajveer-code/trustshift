"""Phase 3C / T3.2 S1 (docs/PHASE3_DESIGN.md ## T3.2): within-CIC-DDoS2019 mechanism-coverage
holdout severity ladder. Binary benign-vs-attack task throughout -- S1 NEVER constructs a
multiclass task (S2 is NOT APPLICABLE, see results/security_S2_feasibility.json). Terminology
(locked): "novel attack-mechanism / unseen attack-family deployment shift", never "novel-class
shift" or "label-space expansion" -- the binary label space never changes; what changes is the
conditional feature structure of the positive class.

Reuses _load_parquet_dir/harmonize/to_binary/_family/_cap/SAMPLE_N_PER_CLASS from
domains/security/run_pipeline.py (same re-orchestration pattern domains/security/
provenance_audit.py already uses -- these are separately-importable primitives; only the
monolithic build() orchestration is not reused, by design, since S1 needs a different training
loop over held-out-family conditions). Uses the domain's PRIMARY model (results/
primary_models.json: security -> xgboost) with the exact same hyperparameters run_pipeline.py
uses, so S1 numbers are comparable to every other security number in the benchmark.

"Matched training-set-size reduction" for the negative control is computed POST-cap (not on raw
row counts): the baseline (full-coverage, severity=0) capped training set is built once per seed,
then the holdout condition's capped size is subtracted from it to get the real row deficit, and
that many rows are removed from the single largest trainable reflection_volumetric family (drawn
from the SAME baseline capped set) for the matched negative control -- this is the only way the
two conditions end up with truly equal total training-set size once SAMPLE_N_PER_CLASS capping is
applied to both.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import P, SEEDS_NEW, SEED  # noqa: E402
from domains.security.run_pipeline import (  # noqa: E402
    _load_parquet_dir, harmonize, to_binary, _family, _cap, SAMPLE_N_PER_CLASS,
)

TRAINABLE_REFLECTION = ["ldap", "mssql", "netbios", "portmap", "udp"]  # frozen, PHASE3_DESIGN.md T3.2
SEVERITY_HOLDOUT_COUNT = {"low": 1, "medium": 2, "high": 3}


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _recall_with_ci(pred: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "recall": None, "ci95": [None, None]}
    k = int(pred[mask].sum())
    lo, hi = _clopper_pearson(k, n)
    return {"n": n, "recall": k / n, "ci95": [lo, hi]}


def load():
    import truststore
    truststore.inject_into_ssl()
    import kagglehub
    p19 = kagglehub.dataset_download("dhoogla/cicddos2019")
    p17 = kagglehub.dataset_download("dhoogla/cicids2017")
    df19, lab19 = harmonize(_load_parquet_dir(p19))
    df17, lab17 = harmonize(_load_parquet_dir(p17))
    num19 = df19.drop(columns=[lab19]).select_dtypes("number").columns
    num17 = df17.drop(columns=[lab17]).select_dtypes("number").columns
    FEATURES = sorted(set(num19) & set(num17))
    df19["_ybin"] = to_binary(df19[lab19]); df19["_fam"] = _family(df19[lab19])
    df17["_ybin"] = to_binary(df17[lab17]); df17["_fam"] = _family(df17[lab17])
    low17 = df17[lab17].astype(str).str.lower()
    df17 = df17[(low17.isin(["benign", "normal"])) | (df17["_ybin"] == 1)].reset_index(drop=True)
    is_train_file = df19["_srcfile"].str.endswith("-training")
    is_test_file = df19["_srcfile"].str.endswith("-testing")
    df19_train_raw = df19[is_train_file].reset_index(drop=True)
    df19_test_raw = df19[is_test_file].reset_index(drop=True)
    return df19_train_raw, df19_test_raw, df17, FEATURES


def _severity_order(df19_train_raw: pd.DataFrame) -> list[str]:
    """Fixed rule: ascending native-training-file row count, smallest first -- reproducible from
    a rule, not selected after seeing results (PHASE3_DESIGN.md T3.2)."""
    counts = df19_train_raw["_fam"].value_counts()
    return sorted(TRAINABLE_REFLECTION, key=lambda f: counts.get(f, 0))


def _fit_and_eval(train_capped, test_in_source_c, test_target_c, FEATURES, seed, held_out_families):
    med = train_capped[FEATURES].median()
    Xtr = train_capped[FEATURES].fillna(med).to_numpy(np.float32)
    ytr = train_capped["_ybin"].to_numpy()
    scaler = MinMaxScaler().fit(Xtr)
    clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8,
                        colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                        random_state=seed, n_jobs=-1, verbosity=0)
    clf.fit(scaler.transform(Xtr), ytr, sample_weight=compute_sample_weight("balanced", ytr))

    def _eval(d):
        X = np.clip(scaler.transform(d[FEATURES].fillna(med).to_numpy(np.float32)), 0, 1)
        y = d["_ybin"].to_numpy(); fam = d["_fam"].to_numpy()
        p = clf.predict_proba(X)[:, 1]
        pred = (p >= 0.5).astype(int)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
        attack_mask = y == 1
        seen_mask = attack_mask & ~np.isin(fam, held_out_families)
        unseen_mask = attack_mask & np.isin(fam, held_out_families)
        return {"n": int(len(d)), "auc": auc,
                "seen_family_recall": _recall_with_ci(pred, seen_mask),
                "unseen_family_recall": _recall_with_ci(pred, unseen_mask),
                "overall_attack_recall": _recall_with_ci(pred, attack_mask),
                "per_family_recall": {str(f): _recall_with_ci(pred, attack_mask & (fam == f))
                                      for f in sorted(set(fam[attack_mask]))}}
    return {"in_source_test": _eval(test_in_source_c), "target_cicids2017": _eval(test_target_c)}


def run():
    df19_train_raw, df19_test_raw, df17, FEATURES = load()
    df19_test_c = _cap(df19_test_raw, "_fam", SAMPLE_N_PER_CLASS, SEED)
    df17_c = _cap(df17, "_fam", SAMPLE_N_PER_CLASS, SEED)
    order = _severity_order(df19_train_raw)
    counts = df19_train_raw["_fam"].value_counts()
    biggest_fam = order[-1]
    print(f"severity holdout order (ascending row count): {[(f, int(counts.get(f, 0))) for f in order]}")

    results = {"trainable_reflection_families": order,
               "family_row_counts": {f: int(counts.get(f, 0)) for f in order},
               "biggest_family_for_negative_control": biggest_fam,
               "baseline": {"per_seed": []}, "severity_levels": {}, "negative_controls": {}}

    baseline_capped = {}
    for seed in SEEDS_NEW:
        tc = _cap(df19_train_raw, "_fam", SAMPLE_N_PER_CLASS, seed)
        baseline_capped[seed] = tc
        ev = _fit_and_eval(tc, df19_test_c, df17_c, FEATURES, seed, [])
        results["baseline"]["per_seed"].append({"seed": seed, "n_train": int(len(tc)), **ev})
    print(f"baseline (full coverage): n_train per seed = "
          f"{[e['n_train'] for e in results['baseline']['per_seed']]}")

    for level, k in SEVERITY_HOLDOUT_COUNT.items():
        held_out = order[:k]
        per_seed, nc_per_seed = [], []
        for seed in SEEDS_NEW:
            full_capped = baseline_capped[seed]
            holdout_train = df19_train_raw[~df19_train_raw["_fam"].isin(held_out)]
            holdout_capped = _cap(holdout_train, "_fam", SAMPLE_N_PER_CLASS, seed)
            n_removed = len(full_capped) - len(holdout_capped)
            ev = _fit_and_eval(holdout_capped, df19_test_c, df17_c, FEATURES, seed, held_out)
            per_seed.append({"seed": seed, "n_train": int(len(holdout_capped)),
                             "n_removed_vs_baseline": int(n_removed), **ev})

            fam_rows = full_capped[full_capped["_fam"] == biggest_fam]
            n_drop = min(n_removed, len(fam_rows))
            drop_idx = fam_rows.sample(n=n_drop, random_state=seed).index if n_drop > 0 else fam_rows.index[:0]
            nc_train = full_capped.drop(index=drop_idx)
            nc_ev = _fit_and_eval(nc_train, df19_test_c, df17_c, FEATURES, seed, [])
            nc_per_seed.append({"seed": seed, "n_train": int(len(nc_train)),
                                "n_removed_vs_baseline": int(n_drop), **nc_ev})
        results["severity_levels"][level] = {
            "held_out_families": held_out, "n_held_out": k,
            "coverage_reduction_frac": k / len(TRAINABLE_REFLECTION), "per_seed": per_seed}
        results["negative_controls"][level] = {
            "matched_to_severity": level, "subsampled_family": biggest_fam, "per_seed": nc_per_seed}
        avg_removed = np.mean([p["n_removed_vs_baseline"] for p in per_seed])
        print(f"{level} (hold out {held_out}): avg rows removed vs baseline = {avg_removed:.0f}; "
              f"negative control matched by subsampling '{biggest_fam}' by the same amount")

    (P["out"] / "phase3_T32_S1_full.json").write_text(json.dumps(results, indent=2))
    _write_ladder_csv(results)
    return results


def _write_ladder_csv(results):
    rows = []
    for kind, block in (("baseline", {"baseline": results["baseline"]}),
                        ("mechanism_holdout", results["severity_levels"]),
                        ("negative_control", results["negative_controls"])):
        for level, entry in block.items():
            for ps in entry["per_seed"]:
                for split in ("in_source_test", "target_cicids2017"):
                    e = ps[split]
                    rows.append({
                        "condition": kind, "severity": level, "seed": ps["seed"],
                        "n_train": ps["n_train"], "split": split, "n_eval": e["n"], "auc": e["auc"],
                        "seen_recall": e["seen_family_recall"]["recall"],
                        "seen_recall_n": e["seen_family_recall"]["n"],
                        "seen_recall_ci_lo": e["seen_family_recall"]["ci95"][0],
                        "seen_recall_ci_hi": e["seen_family_recall"]["ci95"][1],
                        "unseen_recall": e["unseen_family_recall"]["recall"],
                        "unseen_recall_n": e["unseen_family_recall"]["n"],
                        "unseen_recall_ci_lo": e["unseen_family_recall"]["ci95"][0],
                        "unseen_recall_ci_hi": e["unseen_family_recall"]["ci95"][1],
                        "overall_recall": e["overall_attack_recall"]["recall"],
                        "overall_recall_n": e["overall_attack_recall"]["n"],
                        "overall_recall_ci_lo": e["overall_attack_recall"]["ci95"][0],
                        "overall_recall_ci_hi": e["overall_attack_recall"]["ci95"][1],
                    })
    pd.DataFrame(rows).to_csv(P["out"] / "security_mechanism_ladder.csv", index=False)
    print(f"wrote security_mechanism_ladder.csv ({len(rows)} rows)")


if __name__ == "__main__":
    run()
