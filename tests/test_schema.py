import numpy as np
import pandas as pd
import pytest

from schema import validate_predictions


def toy(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "domain": "clinical",
        "model": "xgboost",
        "seed": 42,
        "split": ["source_test"] * (n // 2) + ["target_brfss"] * (n // 2),
        "y_true": rng.integers(0, 2, n),
        "class_label": None,
        "p_hat": rng.uniform(0, 1, n),
        "subgroup_axis": "sex",
        "subgroup": ["male", "female"] * (n // 2),
        "row_id": [f"r{i}" for i in range(n)],
    })


def test_valid_frame_passes():
    assert validate_predictions(toy()) == []


def test_missing_column():
    assert any("missing columns" in e for e in validate_predictions(toy().drop(columns=["p_hat"])))


def test_bad_domain():
    df = toy(); df.loc[0, "domain"] = "astrology"
    assert any("unknown domain" in e for e in validate_predictions(df))


def test_y_true_not_binary():
    df = toy(); df.loc[0, "y_true"] = 2
    assert any("y_true" in e for e in validate_predictions(df))


def test_p_hat_out_of_range():
    df = toy(); df.loc[0, "p_hat"] = 1.5
    assert any("p_hat" in e for e in validate_predictions(df))


def test_bad_split_name():
    df = toy(); df.loc[0, "split"] = "test"
    assert any("split values" in e for e in validate_predictions(df))


def test_nan_in_required():
    df = toy(); df.loc[0, "subgroup"] = None
    assert any("NaNs" in e for e in validate_predictions(df))


def test_single_subgroup_axis():
    df = toy(); df["subgroup"] = "male"
    assert any("<2 subgroups" in e for e in validate_predictions(df))


def test_duplicate_row_id():
    df = toy(); df.loc[1, "row_id"] = "r0"
    assert any("duplicate row_id" in e for e in validate_predictions(df))


def test_empty_frame():
    assert any("empty" in e for e in validate_predictions(toy().iloc[0:0]))
