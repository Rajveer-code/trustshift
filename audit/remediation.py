"""Remediation ladder (PLAN.md §3): what does each fix actually restore?

L0 no fix -> L1 post-hoc recalibration (temperature / Platt / isotonic on a 10% stratified
target calibration split, evaluated on the remaining 90%). L1 is the scientific headline:
recalibration is expected to fix calibration (ECE) while leaving discrimination (AUC) and
subgroup reliability (gap) essentially unchanged.

L2 (importance-weighted retrain) and L3 (small labeled-target refit) are tabular-only and need
the trained model + features; implemented for lending in a follow-up pass. NLP L3 = cite P4's
published fine-tuning gain (+0.216 AUC). This module ships L0+L1 for all ready domains.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P, ECE_BINS, TARGET_CALIB_FRAC, SEED  # noqa: E402
from audit.engine import _jsonable  # noqa: E402
from fairscope.core.calibration import expected_calibration_error, isotonic_recalibrate  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

PRIMARY_MODEL = {"clinical": "fedavg", "nlp": "bert", "lending": "lightgbm_temporal",
                 "security": "lightgbm"}
PRIMARY_AXIS = {"clinical": "age", "nlp": "proxy_class", "lending": "race_black",
                "security": "attack_family"}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _temperature(p_cal, y_cal, p_eval):
    z = _logit(p_cal)
    def nll(T):
        q = 1 / (1 + np.exp(-z / T))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.mean(y_cal * np.log(q) + (1 - y_cal) * np.log(1 - q))
    T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
    return 1 / (1 + np.exp(-_logit(p_eval) / T))


def _platt(p_cal, y_cal, p_eval):
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(p_cal).reshape(-1, 1), y_cal)
    return lr.predict_proba(_logit(p_eval).reshape(-1, 1))[:, 1]


def _isotonic(p_cal, y_cal, p_eval):
    cal = isotonic_recalibrate(p_cal, y_cal)
    # fairscope returns a fitted mapping (IsotonicRegression) or an array; handle both
    if hasattr(cal, "predict"):
        return cal.predict(p_eval)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
    return iso.predict(p_eval)


def _gap_auc(y, p, sub):
    vals = {}
    for g in np.unique(sub):
        m = sub == g
        if len(np.unique(y[m])) == 2:
            vals[g] = roc_auc_score(y[m], p[m])
    return (max(vals.values()) - min(vals.values())) if len(vals) >= 2 else np.nan


def _one_split(sdf: pd.DataFrame, multiclass: bool) -> dict:
    """L0/L1 on one target split's primary-axis rows (or per-class macro for nlp)."""
    rng = np.random.default_rng(SEED)

    def eval_block(y, p, sub):
        n = len(y)
        idx = rng.permutation(n)
        k = max(50, int(TARGET_CALIB_FRAC * n))
        cal, ev = idx[:k], idx[k:]
        out = {"ece_L0": expected_calibration_error(y[ev], p[ev], n_bins=ECE_BINS),
               "auc_L0": roc_auc_score(y[ev], p[ev]) if len(np.unique(y[ev])) == 2 else np.nan,
               "gap_L0": _gap_auc(y[ev], p[ev], sub[ev])}
        for name, fn in [("temperature", _temperature), ("platt", _platt), ("isotonic", _isotonic)]:
            pc = fn(p[cal], y[cal], p[ev])
            out[f"ece_{name}"] = expected_calibration_error(y[ev], pc, n_bins=ECE_BINS)
            out[f"auc_{name}"] = roc_auc_score(y[ev], pc) if len(np.unique(y[ev])) == 2 else np.nan
            out[f"gap_{name}"] = _gap_auc(y[ev], pc, sub[ev])
        return out

    if multiclass:  # macro over class groups
        blocks = [eval_block(g.y_true.values, g.p_hat.values, g.subgroup.values)
                  for _, g in sdf.groupby("class_label")]
        return {k: float(np.nanmean([b[k] for b in blocks])) for k in blocks[0]}
    return eval_block(sdf.y_true.values, sdf.p_hat.values, sdf.subgroup.values)


def remediate(domain: str) -> dict:
    df = pd.read_parquet(P["out"] / f"predictions_{domain}.parquet")
    df = df[df.seed == 42] if 42 in set(df.seed.unique()) else df[df.seed == df.seed.min()]
    model = PRIMARY_MODEL[domain]
    paxis = PRIMARY_AXIS[domain]
    multiclass = df["class_label"].notna().any()
    mdf = df[(df.model == model) & (df.subgroup_axis == paxis)]
    res = {"domain": domain, "model": model, "targets": {}}
    for split in sorted(s for s in mdf.split.unique() if s != "source_test"):
        res["targets"][split] = _one_split(mdf[mdf.split == split], multiclass)
    res["reading"] = "L1 expectation: ece_isotonic << ece_L0 while auc_* ~ auc_L0 and gap_* ~ gap_L0"
    return res


def main():
    for domain in ["clinical", "nlp", "lending", "security"]:
        if not (P["out"] / f"predictions_{domain}.parquet").exists():
            print(f"[skip] {domain}")
            continue
        res = remediate(domain)
        (P["out"] / f"remediation_{domain}.json").write_text(json.dumps(res, indent=2, default=_jsonable))
        print(f"\n=== {domain} remediation (primary={res['model']}) ===")
        for tgt, m in res["targets"].items():
            print(f"  {tgt:22s} ECE {m['ece_L0']:.3f}->iso {m['ece_isotonic']:.3f} | "
                  f"AUC {m['auc_L0']:.3f}->iso {m['auc_isotonic']:.3f} | "
                  f"gap {m['gap_L0']:.3f}->iso {m['gap_isotonic']:.3f}")
    print("\nwrote remediation_*.json")


if __name__ == "__main__":
    main()
