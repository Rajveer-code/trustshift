"""Round-6 finalization: H2a/H2b split artifacts requested directly by the user (items 4, 12, 13).
Companion to synthesis/phase3_h2_scientific_audit.py -- reuses its cluster-bootstrap machinery so
the magnitude-proxy-sensitivity check uses the exact same pre-registered method, not a new one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402
from synthesis.phase3_h2_scientific_audit import (  # noqa: E402
    AXES, AXIS_LABEL, _load_shiftpoints, _cluster_bootstrap_assoc,
)


def _load(name):
    fp = P["out"] / name
    return json.loads(fp.read_text()) if fp.exists() else None


# ---------------------------------------------------------------------------------------------
# item 4: results/mechanism_vs_magnitude_final.csv
# ---------------------------------------------------------------------------------------------
def mechanism_vs_magnitude_final_csv() -> pd.DataFrame:
    mvm = _load("phase3_h2_magnitude_vs_mechanism.json")
    fp = pd.read_csv(P["out"] / "phase3_failure_fingerprint.csv")
    d = fp[fp["auc_available"] == True].copy()  # noqa: E712
    d = d[d["mechanism"] != "concept_natural"]
    d = d[(d["source"] != "T3.2/S1") | (d["mechanism"] == "novel_mechanism")]

    def sev_ordinal(s):
        s = str(s).lower()
        return 0 if s.startswith("low") else 1 if s.startswith("medium") else 2 if s.startswith("high") else np.nan

    d["severity_ordinal"] = d["severity"].map(sev_ordinal)
    d = d.dropna(subset=["severity_ordinal", "auc_delta"])
    d["abs_delta_auc"] = d["auc_delta"].abs()
    y = d["abs_delta_auc"].to_numpy()

    rows = [
        {"model": "M0_magnitude_only", "n": mvm["n_rows_used"], "cv_scheme": "leave-one-out",
         "r2_in": mvm["M0_magnitude_only"]["r2"], "r2_out": mvm["M0_magnitude_only"]["loo_cv_r2"],
         "rmse_out": None, "mechanism_features": "none", "magnitude_feature": "severity_ordinal (within-mechanism rung, 0/1/2)"},
        {"model": "M1_mechanism_only", "n": mvm["n_rows_used"], "cv_scheme": "leave-one-out",
         "r2_in": mvm["M1_mechanism_only"]["r2"], "r2_out": mvm["M1_mechanism_only"]["loo_cv_r2"],
         "rmse_out": None, "mechanism_features": "C(mechanism): covariate/prior/novel_mechanism", "magnitude_feature": "none"},
        {"model": "M2_magnitude_plus_mechanism", "n": mvm["n_rows_used"], "cv_scheme": "leave-one-out",
         "r2_in": mvm["M2_magnitude_plus_mechanism"]["r2"], "r2_out": mvm["M2_magnitude_plus_mechanism"]["loo_cv_r2"],
         "rmse_out": None, "mechanism_features": "C(mechanism)", "magnitude_feature": "severity_ordinal"},
    ]

    # RMSE_out via the same LOO loop (mvm only stored R2_out) -- recompute directly here, once, for all 3 models
    import statsmodels.formula.api as smf
    formulas = {"M0_magnitude_only": "abs_delta_auc ~ severity_ordinal",
                "M1_mechanism_only": "abs_delta_auc ~ C(mechanism)",
                "M2_magnitude_plus_mechanism": "abs_delta_auc ~ severity_ordinal + C(mechanism)"}
    for row in rows:
        formula = formulas[row["model"]]
        sq_errs = []
        for i in range(len(d)):
            train = d.drop(d.index[i])
            test = d.iloc[[i]]
            try:
                m = smf.ols(formula, data=train).fit()
                pred = float(m.predict(test).iloc[0])
            except Exception:
                continue
            sq_errs.append((float(test["abs_delta_auc"].iloc[0]) - pred) ** 2)
        row["rmse_out"] = float(np.sqrt(np.mean(sq_errs))) if sq_errs else None

    df = pd.DataFrame(rows)
    df.to_csv(P["out"] / "mechanism_vs_magnitude_final.csv", index=False)
    conclusion = ("The current cross-dataset controlled evidence is insufficient to establish that "
                 "mechanism outperforms magnitude as an out-of-sample predictor.")
    all_negative_out = bool((df["r2_out"] < 0).all())
    (P["out"] / "mechanism_vs_magnitude_final_conclusion.json").write_text(json.dumps({
        "conclusion": conclusion, "all_models_negative_r2_out": all_negative_out,
        "note": "This sentence is retained verbatim from round-6 unless a future, better-powered "
                "experiment changes it (item 4). Do not soften it while all_models_negative_r2_out "
                "is true.",
    }, indent=2))
    print(f"mechanism_vs_magnitude_final_csv: wrote mechanism_vs_magnitude_final.csv "
          f"(all R2_out negative: {all_negative_out})")
    return df


# ---------------------------------------------------------------------------------------------
# items 12/13: magnitude-proxy sensitivity -- does H2a survive using |delta_pi| instead of auc_dc?
# ---------------------------------------------------------------------------------------------
def magnitude_proxy_sensitivity_csv() -> pd.DataFrame:
    df = _load_shiftpoints().copy()
    df["abs_delta_pi"] = df["delta_pi"].abs()

    rows = []
    for proxy_col, proxy_name in (("auc_dc", "AUC_dc"), ("abs_delta_pi", "|Delta-pi|")):
        for axis in AXES:
            r = _cluster_bootstrap_assoc(df, proxy_col, axis, "state_fips", n_boot=500)
            rows.append({
                "domain": "lending", "target": "T3.1 (447 state,year,seed cells, all years/seeds pooled)",
                "magnitude_proxy": proxy_name, "endpoint": AXIS_LABEL[axis],
                "r2": r["point_estimate"]["r2"], "pearson_r": r["point_estimate"]["pearson_r"],
                "spearman_rho": r["point_estimate"]["spearman_rho"],
                "r2_ci95_lo": None, "r2_ci95_hi": None,  # bootstrap CI computed on slope/rho, not r2 directly -- see full artifact
                "slope_ci95": r["cluster_bootstrap_ci95"]["slope"],
                "out_of_sample_metric": "not computed here -- see state_identity_generalization.json "
                                        "for the out-of-sample layer; this table is the H2a "
                                        "in-sample-association robustness check specifically",
            })
    out = pd.DataFrame(rows)
    out.to_csv(P["out"] / "magnitude_proxy_sensitivity.csv", index=False)

    pivot = out.pivot(index="endpoint", columns="magnitude_proxy", values="r2")
    both_weak = bool((pivot.max(axis=1) < 0.10).all())
    scope_note = ("SCOPE LIMIT, stated rather than silently narrowed: only AUC_dc and |Delta-pi| "
                 "are compared here -- both are already-computed columns in "
                 "lending_shiftpoints.csv. PSI (population stability index) and a raw "
                 "score-distribution-drift statistic are NOT currently computed anywhere in this "
                 "codebase and would require new instrumentation; they are not fabricated or "
                 "approximated here.")
    conclusion = ("No single marginal shift proxy (of the two tested) consistently predicted the "
                 "observed failure fingerprint." if both_weak else
                 "At least one endpoint shows a magnitude proxy with R2>=0.10 -- H2a's "
                 "'insufficient regardless of proxy' framing needs the per-endpoint exception "
                 "stated explicitly, not asserted as universal.")
    (P["out"] / "magnitude_proxy_sensitivity_summary.json").write_text(json.dumps({
        "both_proxies_weak_every_endpoint": both_weak, "conclusion": conclusion, "scope_note": scope_note,
        "r2_table": pivot.to_dict(),
    }, indent=2, default=str))
    print(f"magnitude_proxy_sensitivity_csv: wrote magnitude_proxy_sensitivity.csv "
          f"(both proxies weak everywhere: {both_weak})")
    return out


def run():
    mechanism_vs_magnitude_final_csv()
    magnitude_proxy_sensitivity_csv()
    print("phase3_h2ab_finalize.run() complete")


if __name__ == "__main__":
    run()
