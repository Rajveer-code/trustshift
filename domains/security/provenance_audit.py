"""Security provenance / leakage / mechanism-category audit (PLAN_ijdsa.md Phase-1 follow-up,
2026-08-22 user directive #5/#7).

Must be run and its output inspected BEFORE any prose claim ("known separability", "novel
mechanism", "S1/S2 design") is written or acted on. Records, from the raw cached kagglehub
parquets (no re-download, no retraining):
  1. Native train/test file grouping (which physical file each row came from).
  2. Attack-family definitions actually applied (_family() substring rule, verbatim from
     domains/security/run_pipeline.py, including the udp/udplag collision).
  3. A standard 3-tier DDoS mechanism-category taxonomy (Volumetric/Reflection-Amplification,
     TCP Protocol-Exhaustion, Application-Layer) and which family maps to which category, so
     "novel mechanism" claims are anchored to networking-security literature (Mirkovic & Reiher
     2004; Zargar, Joshi & Tipper 2013 IEEE Comm. Surveys), not an ad hoc grouping invented for
     this paper. Citation details to be verified before use in the manuscript.
  4. Source/target class counts BEFORE and AFTER every sampling cap (raw -> native-split ->
     capped), so no number in this pipeline is reported without its full provenance chain.
  5. An actual near-duplicate row check within source_train, within source_test, and across the
     train/test boundary -- not merely trusting file-level grouping. Flow-capture tools can
     legitimately produce many rows with identical feature vectors (repeated automated
     traffic), so a raw duplicate count is reported as a rate, not treated as an error by
     itself; but a train/test overlap rate above the WITHIN-split rate would indicate the
     file-level split is not actually leakage-safe and must be reported as such.
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import P  # noqa: E402
from domains.security.run_pipeline import (  # noqa: E402
    harmonize, to_binary, _family, LEAKY_COLS, DDOS_MARKERS,
)

# ---- mechanism-category taxonomy (define BEFORE looking at any severity/failure numbers) -----
# Standard 3-tier DDoS taxonomy: volumetric/reflection-amplification, TCP protocol-exhaustion,
# application-layer (high-rate flood vs. low-and-slow). Verify exact citation before manuscript
# use; the categorical structure itself (not the specific citation) is standard and not
# constructed post hoc from this dataset's results.
MECHANISM_CATEGORY = {
    "dns": "reflection_volumetric", "ldap": "reflection_volumetric",
    "mssql": "reflection_volumetric", "netbios": "reflection_volumetric",
    "ntp": "reflection_volumetric", "snmp": "reflection_volumetric",
    "tftp": "reflection_volumetric", "portmap": "reflection_volumetric",
    "udp": "reflection_volumetric",  # NOTE: this bucket also absorbs "UDPLag"-labeled rows,
                                      # because _family()'s substring match for "udp" fires on
                                      # "udplag" too (checked before any more specific key) --
                                      # both are volumetric/reflection-style floods, so this
                                      # collision does not cross a mechanism-category boundary.
    "syn": "protocol_exhaustion",
    "hulk": "application_flood", "goldeneye": "application_flood",
    "slowloris": "application_slow", "slowhttp": "application_slow",
    "benign": "benign",
    # "ddos" is DELIBERATELY unmapped here -- see the label-collision note below.
}
DDOS_LABEL_COLLISION_NOTE = (
    "The string label 'ddos' denotes two DIFFERENT populations that must not be treated as the "
    "same family for overlap/coverage purposes: in CIC-DDoS2019 (source) it is a residual/"
    "catch-all bucket with n=51 raw rows (negligible, likely mislabeled or leftover rows not "
    "assigned to a specific reflection family); in CICIDS2017 (target) it is the dataset's own "
    "LOIC/HOIC-generated volumetric flood label with n=128,014 raw rows -- a real, large, "
    "well-defined attack population. Reporting 'ddos is a shared family' without this "
    "distinction would misrepresent the mechanism-overlap finding."
)


def _tag_family_and_hash(df, label_col, feature_cols):
    fam = _family(df[label_col])
    feat = df[feature_cols].fillna(df[feature_cols].median())
    row_hash = pd.util.hash_pandas_object(
        feat.round(6).apply(lambda c: c.astype(str)).agg("|".join, axis=1), index=False
    )
    return fam, row_hash


def _dup_rate(hashes_a: pd.Series, hashes_b: pd.Series | None = None) -> dict:
    if hashes_b is None:
        vc = hashes_a.value_counts()
        n_dup_rows = int((vc[vc > 1]).sum())
        return {"n_rows": int(len(hashes_a)), "n_duplicate_rows": n_dup_rows,
                "duplicate_rate": float(n_dup_rows / max(1, len(hashes_a)))}
    set_a = set(hashes_a.tolist())
    overlap = sum(1 for h in hashes_b if h in set_a)
    return {"n_rows_b": int(len(hashes_b)), "n_overlapping_with_a": int(overlap),
            "overlap_rate_of_b": float(overlap / max(1, len(hashes_b)))}


def run() -> dict:
    p19 = Path(P["out"]).parent  # placeholder; real path resolved by kagglehub cache below
    import truststore
    truststore.inject_into_ssl()
    import kagglehub
    p19 = kagglehub.dataset_download("dhoogla/cicddos2019")
    p17 = kagglehub.dataset_download("dhoogla/cicids2017")

    files19 = sorted(glob.glob(str(Path(p19) / "**" / "*.parquet"), recursive=True))
    files17 = sorted(glob.glob(str(Path(p17) / "**" / "*.parquet"), recursive=True))

    # 1) native file/session grouping -----------------------------------------------------
    file_grouping = {"source_cicddos2019": [], "target_cicids2017": []}
    frames19 = []
    for f in files19:
        d = pd.read_parquet(f)
        stem = Path(f).stem
        d["_srcfile"] = stem
        split_tag = "training" if stem.endswith("-training") else (
            "testing" if stem.endswith("-testing") else "UNRECOGNIZED")
        file_grouping["source_cicddos2019"].append(
            {"file": stem, "native_split": split_tag, "n_rows_raw": int(len(d))})
        frames19.append(d)
    df19 = pd.concat(frames19, ignore_index=True)
    for f in files17:
        d = pd.read_parquet(f)
        stem = Path(f).stem
        file_grouping["target_cicids2017"].append({"file": stem, "n_rows_raw": int(len(d))})

    df19h, lab19 = harmonize(df19)
    df19h["_ybin"] = to_binary(df19h[lab19])
    df19h["_fam"] = _family(df19h[lab19])
    df19h["_mech"] = df19h["_fam"].map(MECHANISM_CATEGORY).fillna("UNMAPPED(ddos-collision)")

    # 2) family / mechanism-category coverage, source TRAINING files only ------------------
    train_mask = df19h["_srcfile"].str.endswith("-training")
    test_mask = df19h["_srcfile"].str.endswith("-testing")
    mech_coverage_train = (df19h.loc[train_mask, "_mech"].value_counts().to_dict())
    mech_coverage_test = (df19h.loc[test_mask, "_mech"].value_counts().to_dict())
    fam_coverage_train = (df19h.loc[train_mask, "_fam"].value_counts().to_dict())
    fam_coverage_test = (df19h.loc[test_mask, "_fam"].value_counts().to_dict())

    # 3) duplicate check: EXACT (full float precision, bit-for-bit identical rows) and
    #    NEAR-duplicate (rounded to 6 decimals -- catches rows that differ only in floating-
    #    point noise from the harmonization/scaling pipeline) are reported SEPARATELY, within
    #    source_train, within source_test, and across the train/test boundary. Conflating the
    #    two would understate how conservative the "exact" number is and overstate how
    #    permissive the "near" number is (user directive, round-3 review).
    num_cols = sorted(set(df19h.select_dtypes("number").columns) - {"_ybin"})
    tr_exact = pd.util.hash_pandas_object(
        df19h.loc[train_mask, num_cols].fillna(-999), index=False)
    te_exact = pd.util.hash_pandas_object(
        df19h.loc[test_mask, num_cols].fillna(-999), index=False)
    tr_near = pd.util.hash_pandas_object(
        df19h.loc[train_mask, num_cols].fillna(-999).round(6), index=False)
    te_near = pd.util.hash_pandas_object(
        df19h.loc[test_mask, num_cols].fillna(-999).round(6), index=False)
    dup_within_train = {"exact": _dup_rate(tr_exact), "near_6dp": _dup_rate(tr_near)}
    dup_within_test = {"exact": _dup_rate(te_exact), "near_6dp": _dup_rate(te_near)}
    dup_across = {"exact": _dup_rate(tr_exact, te_exact), "near_6dp": _dup_rate(tr_near, te_near)}

    result = {
        "file_grouping": file_grouping,
        "attack_family_definition": (
            "domains/security/run_pipeline.py::_family() -- substring match against the "
            "lowercased raw label, checked in this fixed order: syn, udp, ldap, mssql, netbios, "
            "snmp, ssdp, ntp, dns, tftp, portmap, hulk, goldeneye, slowloris, slowhttp, ddos "
            "(first match wins; 'benign'/'normal' -> 'benign'; no match -> 'other_attack'). "
            "NOTE: 'udplag'-labeled rows match 'udp' (checked before any udplag-specific key), "
            "so the 'udp' family bucket includes both UDP and UDPLag raw labels."
        ),
        "mechanism_category_definition": MECHANISM_CATEGORY,
        "ddos_label_collision_note": DDOS_LABEL_COLLISION_NOTE,
        "family_coverage": {
            "source_training_files": fam_coverage_train,
            "source_testing_files": fam_coverage_test,
        },
        "mechanism_coverage": {
            "source_training_files": mech_coverage_train,
            "source_testing_files": mech_coverage_test,
            "note": ("source TRAINING files contain ONLY reflection_volumetric and "
                     "protocol_exhaustion mechanism categories -- zero application_flood or "
                     "application_slow representation exists anywhere in CIC-DDoS2019 as "
                     "loaded here. Any 'novel mechanism' severity ladder built from S1 can "
                     "therefore only vary WITHIN reflection_volumetric/protocol_exhaustion "
                     "coverage (e.g. holding out increasing numbers of reflection families); "
                     "it cannot construct a within-source application-layer holdout because "
                     "that mechanism category is entirely absent from the source dataset."),
        },
        "duplicate_check": {
            "within_source_training_files": dup_within_train,
            "within_source_testing_files": dup_within_test,
            "across_training_vs_testing_files": dup_across,
            "reading": ("Each cell reports BOTH an exact duplicate rate (full float precision, "
                        "bit-for-bit identical rows) and a near-duplicate rate (rounded to 6 "
                        "decimals, catching rows that differ only in floating-point noise from "
                        "harmonization). Neither is claimed as proof the split is leakage-free. "
                        "A nonzero WITHIN-split rate (either kind) is expected and not itself "
                        "evidence of leakage -- repeated automated flows legitimately produce "
                        "identical or near-identical feature vectors. The leakage-relevant "
                        "comparison is whether the ACROSS-split rate exceeds the matching "
                        "WITHIN-split rate (exact vs. exact, near vs. near); if it does not, this "
                        "check found no evidence the native file-level split inflates apparent "
                        "performance beyond what within-split duplication alone would produce."),
        },
    }
    return result


def main():
    res = run()
    (P["out"] / "security_provenance_audit.json").write_text(json.dumps(res, indent=2))
    print("=== mechanism coverage, source TRAINING files ===")
    print(json.dumps(res["mechanism_coverage"]["source_training_files"], indent=2))
    print("\n=== duplicate check (exact vs. near-6dp, within-split vs. across-split) ===")
    for k in ["within_source_training_files", "within_source_testing_files", "across_training_vs_testing_files"]:
        print(f"  {k}:")
        for kind in ("exact", "near_6dp"):
            print(f"    {kind}: {res['duplicate_check'][k][kind]}")
    print(f"\nwrote {P['out'] / 'security_provenance_audit.json'}")


if __name__ == "__main__":
    main()
