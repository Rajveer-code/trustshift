"""Cross-domain META-ANALYSIS (PLAN.md §3): the intellectual centerpiece.

Every (domain x model x target) is one shift-point. We ask whether trustworthiness failure is
PREDICTABLE across domains from cheap diagnostics — reframing the paper from evaluation to
prediction (exploratory cross-domain association; no causal claim; small n stated openly):

  M1: delta_gap_auc ~ delta_ece         (does calibration decay predict subgroup-reliability decay?)
  M2: delta_gap_auc ~ auc_dc            (does more covariate shift mean more subgroup loss?)
  M3: delta_gap_auc ~ delta_auc         (does aggregate-accuracy loss under-predict subgroup loss?)

Each fit by bootstrap OLS (resample shift-points, N=2000): slope, 95% CI, R^2. Effect sizes and
CIs are reported; p-values are de-emphasized (with 1.28M clinical rows every p->0).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P, N_BOOT, SEED  # noqa: E402
from audit.engine import _jsonable  # noqa: E402

DOMAINS = ["clinical", "nlp", "lending", "security"]


def _load(domain: str):
    fp = P["out"] / f"audit_{domain}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def assemble_points() -> pd.DataFrame:
    """One row per (domain, model, target) shift-point on the primary seed."""
    rows = []
    for domain in DOMAINS:
        a = _load(domain)
        if not a:
            continue
        diag = P["out"] / f"diagnosis_{domain}.json"
        diag = json.loads(diag.read_text()) if diag.exists() else {}
        cov = diag.get("covariate", {})
        for model, seeds in a["models"].items():
            s = seeds.get("42") or next(iter(seeds.values()))
            src_ece = s["splits"]["source_test"]["ece"]
            for split, d in s["deltas"].items():
                tgt_ece = s["splits"][split]["ece"]
                auc_dc = None
                c = cov.get(split)
                if isinstance(c, dict):
                    auc_dc = c.get("auc_dc")
                rows.append({
                    "domain": domain, "model": model, "target": split,
                    "delta_auc": d["delta_auc"],
                    "delta_ece": tgt_ece - src_ece,          # positive => calibration worsened
                    "delta_gap_auc": d.get("delta_gap_auc"),
                    "auc_dc": auc_dc,
                })
    return pd.DataFrame(rows)


def _boot_ols(x: np.ndarray, y: np.ndarray, n_boot=N_BOOT, seed=SEED):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return dict(n=int(len(x)), slope=None, ci=[None, None], r2=None, note="too few points")
    rng = np.random.default_rng(seed)
    def fit(xi, yi):
        b1, b0 = np.polyfit(xi, yi, 1)
        yhat = b1 * xi + b0
        ss = 1 - np.sum((yi - yhat) ** 2) / max(np.sum((yi - yi.mean()) ** 2), 1e-12)
        return b1, ss
    slope0, r2 = fit(x, y)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        try:
            slopes.append(fit(x[idx], y[idx])[0])
        except Exception:
            pass
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return dict(n=int(len(x)), slope=float(slope0), ci=[float(lo), float(hi)],
                r2=float(r2), sig=bool(lo > 0 or hi < 0))


def run() -> dict:
    pts = assemble_points()
    pts.to_csv(P["out"] / "meta_shift_points.csv", index=False)
    g = pts["delta_gap_auc"].to_numpy(float)
    res = {
        "n_shift_points": int(len(pts)),
        "domains_present": sorted(pts.domain.unique().tolist()),
        "M1_gap_vs_calibration": _boot_ols(pts["delta_ece"].to_numpy(float), g),
        "M2_gap_vs_covariateshift": _boot_ols(pts["auc_dc"].to_numpy(float), g),
        "M3_gap_vs_accuracyloss": _boot_ols(pts["delta_auc"].to_numpy(float), g),
    }
    return res


def main():
    res = run()
    (P["out"] / "meta_analysis.json").write_text(json.dumps(res, indent=2, default=_jsonable))
    print(f"shift-points: {res['n_shift_points']}  domains: {res['domains_present']}")
    for k in ("M1_gap_vs_calibration", "M2_gap_vs_covariateshift", "M3_gap_vs_accuracyloss"):
        m = res[k]
        print(f"  {k:26s} n={m['n']} slope={m.get('slope')} CI={m.get('ci')} R2={m.get('r2')}")
    if res["n_shift_points"] < 25:
        print("  [note] <25 shift-points — report as exploratory; add security to strengthen")


if __name__ == "__main__":
    main()
