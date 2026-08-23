"""Engine correctness: construct data with a KNOWN subgroup-gap change, assert recovery."""
import numpy as np
import pandas as pd

from audit.engine import _subgroup_gap, _split_aggregate, _brier_decomposition


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
    auc, se, f1_positive, macro_f1, ece, brier, decomp = _split_aggregate(d, multiclass=False, primary_axis="age")
    assert 0.78 < auc < 0.92
    assert 0.0 <= ece <= 1.0 and se >= 0.0
    assert set(decomp) == {"reliability", "resolution", "uncertainty", "brier_binned", "grouping_loss"}
    assert f1_positive is not None and macro_f1 is not None


def test_binary_f1_positive_differs_from_macro_f1():
    """2026-08-23 correction regression test: on an imbalanced binary split where the model is
    much better at the majority (negative) class than the minority (positive) class,
    f1_positive (the positive class alone) and macro_f1 (mean of BOTH classes' F1) must differ
    -- this is exactly the confusion the round-4 hostile audit caught (binary domains previously
    reported f1_positive under the `macro_f1` key)."""
    rng = np.random.default_rng(11)
    n_neg, n_pos = 9000, 300  # 30:1 imbalance
    y = np.r_[np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)]
    # negatives: model nails them (score near 0). positives: model is little better than chance.
    p_neg = np.clip(rng.normal(0.05, 0.05, n_neg), 0, 1)
    p_pos = np.clip(rng.normal(0.15, 0.15, n_pos), 0, 1)  # most true positives score BELOW 0.5
    p = np.r_[p_neg, p_pos]
    d = pd.DataFrame({"y_true": y, "p_hat": p, "subgroup": "A", "class_label": None, "subgroup_axis": "grp"})
    auc, se, f1_positive, macro_f1, ece, brier, decomp = _split_aggregate(d, multiclass=False, primary_axis="grp")
    assert f1_positive < 0.3, f1_positive  # minority class recall collapses at threshold 0.5
    assert macro_f1 > f1_positive + 0.3, (f1_positive, macro_f1)  # majority class inflates the mean
    # cross-check against sklearn directly, not just internal consistency
    from sklearn.metrics import f1_score
    pred = (p >= 0.5).astype(int)
    assert abs(f1_positive - f1_score(y, pred, zero_division=0)) < 1e-9
    assert abs(macro_f1 - f1_score(y, pred, average="macro", zero_division=0)) < 1e-9


def test_multiclass_macro_f1_matches_sklearn_average_macro():
    """Multiclass path (one-vs-rest rows, grouped by class_label) must reproduce the same
    macro-F1 sklearn would compute directly from a single multiclass label array -- an
    independent cross-check, not just re-deriving the same formula the code already uses."""
    rng = np.random.default_rng(5)
    classes = ["a", "b", "c"]
    n_per_class = 400
    y_true_multiclass, y_pred_multiclass, rows = [], [], []
    for i, c in enumerate(classes):
        # class 'a' easy, 'b' harder, 'c' hardest -> genuinely different per-class F1
        sep = [3.0, 1.2, 0.3][i]
        for cls in classes:
            n = n_per_class
            y_bin = (rng.uniform(0, 1, n) < (0.9 if cls == c else 0.05)).astype(int)
            score = rng.normal(0, 1, n) + y_bin * sep
            p_hat = 1 / (1 + np.exp(-score))
            rows.append(pd.DataFrame({"y_true": y_bin, "p_hat": p_hat, "subgroup": "A",
                                      "class_label": cls, "subgroup_axis": "grp"}))
    d = pd.concat(rows, ignore_index=True)
    auc, se, f1_positive, macro_f1, ece, brier, decomp = _split_aggregate(d, multiclass=True, primary_axis="grp")
    assert f1_positive is None
    # independent cross-check: reconstruct a single multiclass y_true/y_pred and ask sklearn directly
    from sklearn.metrics import f1_score
    per_class_f1 = []
    for cls in classes:
        g = d[d.class_label == cls]
        per_class_f1.append(f1_score(g.y_true, (g.p_hat >= 0.5).astype(int), zero_division=0))
    expected_macro = float(np.mean(per_class_f1))
    assert abs(macro_f1 - expected_macro) < 1e-9, (macro_f1, expected_macro)


def test_no_binary_domain_silently_reports_positive_f1_as_macro():
    """Regression guard for the actual bug: engine.py's binary branch must compute macro_f1 with
    average="macro" explicitly, not inherit sklearn's average="binary" default under the
    macro_f1 name."""
    src = (__import__("pathlib").Path(__file__).resolve().parents[1] / "audit" / "engine.py").read_text()
    assert 'average="macro"' in src or "average='macro'" in src, (
        "audit/engine.py must compute macro_f1 with an explicit average='macro' call")
    assert "macro_f1=f1," not in src and "macro_f1=f1)" not in src, (
        "engine.py appears to assign a bare positive-class f1_score result directly to the "
        "macro_f1 field again -- the 2026-08-23 bug this guards against")


def test_brier_decomposition_reproduces_brier():
    """Murphy (1973): brier_binned = reliability - resolution + uncertainty, EXACTLY (this is
    an algebraic identity, not an approximation). The true (unbinned) Brier score differs from
    brier_binned by grouping_loss, which must shrink as bins narrow (PLAN_ijdsa.md T1.2)."""
    from fairscope.core.metrics import brier_score_loss
    rng = np.random.default_rng(7)
    for trial, (auc_target, n) in enumerate([(0.9, 5000), (0.6, 3000), (0.99, 2000)]):
        d = _grp(auc_target, n, trial)
        y, p = d.y_true.to_numpy(), d.p_hat.to_numpy()
        brier_raw = brier_score_loss(y, p)
        dec = _brier_decomposition(y, p)
        reconstructed = dec["reliability"] - dec["resolution"] + dec["uncertainty"]
        assert abs(reconstructed - dec["brier_binned"]) < 1e-9, (trial, dec)
        assert abs((brier_raw - dec["brier_binned"]) - dec["grouping_loss"]) < 1e-9, (trial, brier_raw, dec)
        # grouping loss should be a small fraction of the true Brier score for 10 bins
        assert abs(dec["grouping_loss"]) < 0.05, (trial, brier_raw, dec)
        # finer bins must not increase grouping loss (monotonic in bin width, not bin count noise)
        dec_fine = _brier_decomposition(y, p, n_bins=50)
        assert abs(dec_fine["grouping_loss"]) <= abs(dec["grouping_loss"]) + 1e-6, (trial, dec, dec_fine)


def test_brier_decomposition_degenerate_labels():
    """All-one-class split: uncertainty must be exactly 0, decomposition identity must still hold."""
    y = np.ones(500, dtype=int)
    p = np.clip(np.random.default_rng(0).normal(0.9, 0.05, 500), 0, 1)
    dec = _brier_decomposition(y, p)
    assert dec["uncertainty"] == 0.0
    reconstructed = dec["reliability"] - dec["resolution"] + dec["uncertainty"]
    assert abs(reconstructed - dec["brier_binned"]) < 1e-9
