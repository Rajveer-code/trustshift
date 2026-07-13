"""Engine correctness: construct data with a KNOWN subgroup-gap change, assert recovery."""
import numpy as np
import pandas as pd

from audit.engine import _subgroup_gap, _split_aggregate


def _grp(auc_target: float, n: int, seed: int) -> pd.DataFrame:
    """One subgroup whose score separability yields approximately `auc_target`."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # shift positives up by delta; larger delta -> higher AUC
    delta = (auc_target - 0.5) * 4.0
    p = rng.normal(0, 1, n) + y * delta
    p = (p - p.min()) / (p.max() - p.min())
    return pd.DataFrame({"y_true": y, "p_hat": p})


def test_gap_recovers_known_value():
    # group A ~0.90 AUC, group B ~0.70 AUC => gap ~0.20
    a = _grp(0.90, 8000, 1).assign(subgroup="A")
    b = _grp(0.70, 8000, 2).assign(subgroup="B")
    axis = pd.concat([a, b], ignore_index=True)
    gap, vals = _subgroup_gap(axis, "auc")
    assert 0.12 < gap < 0.28, (gap, vals)
    assert vals["A"] > vals["B"]


def test_aggregate_auc_binary():
    d = _grp(0.85, 6000, 3).assign(subgroup="A", class_label=None, subgroup_axis="age")
    auc, se, f1, ece, brier = _split_aggregate(d, multiclass=False, primary_axis="age")
    assert 0.78 < auc < 0.92
    assert 0.0 <= ece <= 1.0 and se >= 0.0
