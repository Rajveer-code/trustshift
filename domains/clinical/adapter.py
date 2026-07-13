"""Clinical domain adapter: NHANES (source) -> BRFSS (target) diabetes risk.

Reuses saved predictions from the federated-diabetes repo (PLAN.md §2a) — NO retraining.
Provenance of every input array is the diabetes repo's own scripts:
  - pred_{model}_internal.npy / _external.npy : written by federated/07_external_validation.py
  - y_true_internal.npy / y_true_brfss.npy    : same script
  - brfss_age_{elderly,young}.npy             : same script (age masks on BRFSS)
  - centralised_full.csv                       : NHANES rows aligned 1:1 to *_internal arrays
                                                 (DIABETES column == y_true_internal, verified)

Subgroup coverage (honest, PLAN.md §2a): the BRFSS raw table is not on disk, so only the
age axis is available on the *target* side (saved masks). The internal (source) side carries
all four axes from centralised_full.csv. Sex/race/BMI therefore contribute source-side
subgroup metrics; only age yields a cross-split gap change (Delta-G). To add target-side
sex/race/BMI, rebuild BRFSS per PLAN.md §2a and re-run the repo's 07_external_validation.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import P  # noqa: E402
from schema import save_validated  # noqa: E402

REPO = P["clinical_repo"]
RES = REPO / "results"
MODELS = {"fedavg": "primary", "xgb": "secondary"}  # both have _internal and _external arrays


def _age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[-1, 39, 59, 200], labels=["18-39", "40-59", "60+"]).astype(str)


def _bmi_group(bmi: pd.Series) -> pd.Series:
    return pd.cut(bmi, bins=[-1, 24.9, 29.9, 500], labels=["normal", "overweight", "obese"]).astype(str)


def _sex_group(g: pd.Series) -> pd.Series:
    # NHANES RIAGENDR: 1=male, 2=female
    return g.map({1: "male", 2: "female"}).fillna("unknown").astype(str)


def _race_group(r: pd.Series) -> pd.Series:
    # NHANES RIDRETH3 codes; collapse to interpretable labels
    m = {1: "mexican_american", 2: "other_hispanic", 3: "white_nh", 4: "black_nh",
         6: "asian_nh", 7: "other_multi"}
    return r.map(m).fillna("other").astype(str)


def _rows(domain, model, split, y, p, axis, subgroup, id_prefix):
    n = len(y)
    return pd.DataFrame({
        "domain": domain, "model": model, "seed": 42, "split": split,
        "y_true": np.asarray(y).astype(int), "class_label": None,
        "p_hat": np.clip(np.asarray(p, dtype=float), 0.0, 1.0),
        "subgroup_axis": axis, "subgroup": np.asarray(subgroup).astype(str),
        "row_id": [f"{id_prefix}_{i}" for i in range(n)],
    })


def build() -> pd.DataFrame:
    y_int = np.load(RES / "y_true_internal.npy")
    y_ext = np.load(RES / "y_true_brfss.npy")
    cf = pd.read_csv(REPO / "data" / "centralised_full.csv")
    assert (cf["DIABETES"].values == y_int).all(), "centralised_full misaligned with internal preds"

    # BRFSS age from saved masks: elderly(60+), young(18-39), neither -> 40-59
    el = np.load(RES / "brfss_age_elderly.npy")
    yo = np.load(RES / "brfss_age_young.npy")
    brfss_age = np.where(el, "60+", np.where(yo, "18-39", "40-59"))

    internal_axes = {
        "age": _age_group(cf["RIDAGEYR"]),
        "sex": _sex_group(cf["RIAGENDR"]),
        "race": _race_group(cf["RIDRETH3"]),
        "bmi": _bmi_group(cf["BMXBMI"]),
    }

    frames = []
    for model in MODELS:
        p_int = np.load(RES / f"pred_{model}_internal.npy")
        p_ext = np.load(RES / f"pred_{model}_external.npy")
        for axis, groups in internal_axes.items():
            frames.append(_rows("clinical", model, "source_test", y_int, p_int, axis, groups,
                                 f"cl_{model}_int_{axis}"))
        # target side: age axis only (masks available)
        frames.append(_rows("clinical", model, "target_brfss", y_ext, p_ext, "age", brfss_age,
                             f"cl_{model}_ext_age"))
    return pd.concat(frames, ignore_index=True)


def main():
    df = build()
    out = P["out"] / "predictions_clinical.parquet"
    save_validated(df, out)

    from sklearn.metrics import roc_auc_score
    print("\n-- sanity AUCs (recomputed from saved arrays) --")
    for model in MODELS:
        for split in ["source_test", "target_brfss"]:
            sub = df[(df.model == model) & (df.split == split) & (df.subgroup_axis == "age")]
            auc = roc_auc_score(sub.y_true, sub.p_hat)
            print(f"  {model:8s} {split:13s} AUC={auc:.4f}")
    print("  anchor: FedAvg target_brfss should be ~0.757 (published) -> alignment proof")


if __name__ == "__main__":
    main()
