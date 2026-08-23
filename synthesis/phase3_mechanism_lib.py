"""Shared T3.3 construction primitives (docs/PHASE3_DESIGN.md ## T3.3), reused by lending's and
security's per-domain T3.3 driver scripts so the severity-calibration LOGIC is identical across
domains even though the propensity features/strata differ.

Covariate-shift cells: exponential-tilting resample by a propensity score built from named
features, bisection-searched on the tilting temperature until the resample's domain-classifier
AUC (vs. the unresampled pool) hits a target value (0.60 / 0.75 / 0.90 -- PHASE3_DESIGN.md
'Severity levels'). The bisection search uses a CHEAP single train/test split AUC as a fast
proxy (not the full 5-fold out-of-fold estimate) purely to keep the search tractable; once a
temperature is chosen, the FINAL reported auc_dc for that severity level is always the full,
rigorous audit.diagnosis._domain_classifier_auc (5-fold, out-of-fold) -- never the cheap proxy.
If even the most extreme temperature cannot reach target-tol, the cell is marked infeasible at
that severity rather than silently forced (PHASE3_DESIGN.md's explicit failure-condition rule).

Prior/label-shift cells: closed-form subsampling to hit a target |Delta-pi| within a fixed
covariate stratum -- no search needed, exact by construction. If the stratum cannot reach the
target ratio without heavy replacement (effectively duplicating rows), the cell is marked invalid
at that severity (PHASE3_DESIGN.md's explicit failure condition for the prior cell).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.diagnosis import _domain_classifier_auc  # noqa: E402

SEVERITY_AUC_DC_TARGET = {"low": 0.60, "medium": 0.75, "high": 0.90}
SEVERITY_DELTA_PI_TARGET = {"low": 0.10, "medium": 0.25, "high": 0.40}
AUC_DC_TOL = 0.02
MAX_BISECT_ITER = 10
TEMP_MAX = 12.0


def _propensity_score(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Equal-weighted mean of z-scored named columns -- matches PHASE3_DESIGN.md's own example
    ('toward high-LTV/refi-purpose loans'): higher named-feature values => higher propensity."""
    z = np.zeros(len(df))
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        sd = v.std()
        z = z + ((v - v.mean()) / sd if sd > 1e-9 else np.zeros_like(v))
    return z / max(1, len(cols))


def _cheap_auc_dc(Xs: np.ndarray, Xt: np.ndarray, seed: int) -> float:
    """Fast single-split proxy for the bisection search only -- never reported as a final number."""
    X = np.vstack([Xs, Xt]); y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                         random_state=seed, n_jobs=-1, verbose=-1)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def calibrate_covariate_resample(pool_df: pd.DataFrame, propensity_cols: list[str],
                                  feature_cols: list[str], seed: int) -> dict:
    """Bisection-search an exponential-tilting temperature per severity level so the resample's
    (cheap-proxy) AUC_dc vs. the natural pool hits each target; report the FINAL rigorous AUC_dc
    at the chosen temperature. Returns {severity: {resampled_df, achieved_auc_dc, temperature,
    feasible, n}}."""
    prop = _propensity_score(pool_df, propensity_cols)
    natural_X = pool_df[feature_cols].to_numpy(dtype=np.float64)
    n = len(pool_df)
    rng = np.random.default_rng(seed)

    def sample_at(temp: float) -> pd.DataFrame:
        w = np.exp(np.clip(temp * prop, -50, 50))
        w = w / w.sum()
        idx = rng.choice(n, size=n, replace=True, p=w)
        return pool_df.iloc[idx].reset_index(drop=True)

    out = {}
    for level, target in SEVERITY_AUC_DC_TARGET.items():
        hi_sample = sample_at(TEMP_MAX)
        hi_auc = _cheap_auc_dc(natural_X, hi_sample[feature_cols].to_numpy(dtype=np.float64), seed)
        if hi_auc < target - AUC_DC_TOL:
            final_temp, feasible = TEMP_MAX, False
        else:
            lo_t, hi_t, feasible = 0.0, TEMP_MAX, True
            final_temp = TEMP_MAX
            for _ in range(MAX_BISECT_ITER):
                mid = (lo_t + hi_t) / 2
                a = _cheap_auc_dc(natural_X, sample_at(mid)[feature_cols].to_numpy(dtype=np.float64), seed)
                if abs(a - target) <= AUC_DC_TOL:
                    final_temp = mid
                    break
                if a < target:
                    lo_t = mid
                else:
                    hi_t = mid
                final_temp = mid
        resampled = sample_at(final_temp)
        achieved = _domain_classifier_auc(natural_X, resampled[feature_cols].to_numpy(dtype=np.float64), seed=seed)
        out[level] = {"resampled": resampled, "achieved_auc_dc": achieved, "target_auc_dc": target,
                      "temperature": float(final_temp), "feasible": bool(feasible), "n": int(len(resampled))}
    return out


def calibrate_prior_resample(stratum_df: pd.DataFrame, label_col: str, seed: int,
                              max_replacement_frac: float = 0.5) -> dict:
    """Closed-form subsampling within a FIXED stratum to hit each target |Delta-pi|. Positive
    rows are down/up-sampled (never the negative pool, arbitrarily -- whichever class must move
    to hit the target ratio moves); if reaching a target needs replacing more than
    `max_replacement_frac` of a class (i.e. heavy replacement/duplication), that severity is
    marked infeasible (PHASE3_DESIGN.md's explicit failure condition)."""
    y = stratum_df[label_col].astype(int).to_numpy()
    pi0 = float(y.mean())
    n_pos0, n_neg0 = int((y == 1).sum()), int((y == 0).sum())
    rng = np.random.default_rng(seed)
    pos_idx = stratum_df.index[y == 1].to_numpy()
    neg_idx = stratum_df.index[y == 0].to_numpy()

    out = {"natural_pi": pi0, "n_pos_natural": n_pos0, "n_neg_natural": n_neg0}
    for level, delta in SEVERITY_DELTA_PI_TARGET.items():
        for direction, target_pi in (("up", min(0.99, pi0 + delta)), ("down", max(0.01, pi0 - delta))):
            # solve for n_pos at fixed n_neg (keep whichever class is smaller reduction from natural)
            n_neg = n_neg0
            n_pos_needed = int(round(target_pi * n_neg / max(1e-9, (1 - target_pi))))
            replace_frac = abs(n_pos_needed - n_pos0) / max(1, n_pos0)
            feasible = replace_frac <= max_replacement_frac and n_pos_needed >= 10 and n_neg >= 10
            if n_pos_needed <= n_pos0:
                chosen_pos = rng.choice(pos_idx, size=n_pos_needed, replace=False)
            else:
                chosen_pos = rng.choice(pos_idx, size=n_pos_needed, replace=True)
            chosen = np.concatenate([chosen_pos, neg_idx])
            resampled = stratum_df.loc[chosen].reset_index(drop=True)
            achieved_pi = float(resampled[label_col].astype(int).mean())
            achieved_delta_pi = achieved_pi - pi0
            # target_pi is clamped to [0.01, 0.99] above (pi is a proportion); when the natural
            # pi0 is already near that floor/ceiling, the clamped target can sit close to pi0
            # itself, so a "low replacement burden" resample can still fail to reach anywhere
            # near the INTENDED |target_delta_pi| magnitude. Both failure modes -- heavy
            # replacement AND achieved-far-from-intended-target -- must gate feasibility,
            # otherwise a clamped-to-the-floor/ceiling resample is silently reported as a valid
            # severity level (caught in security's syn-mechanism cell: natural_pi=0.010 clamps
            # every "down" target to ~0.01, so achieved_delta_pi stayed ~-0.0003 for low/med/high
            # while target_delta_pi was 0.10/0.25/0.40).
            reaches_target = abs(abs(achieved_delta_pi) - delta) <= 0.02
            feasible = feasible and reaches_target
            key = f"{level}_{direction}"
            out[key] = {"resampled": resampled, "achieved_pi": achieved_pi,
                       "achieved_delta_pi": achieved_delta_pi, "target_delta_pi": delta,
                       "direction": direction, "feasible": bool(feasible),
                       "reaches_target": bool(reaches_target),
                       "replacement_frac": float(replace_frac), "n": int(len(resampled))}
    return out
