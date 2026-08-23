"""Phase 3F negative controls (docs/PHASE3_DESIGN.md T3.3 'Negative controls' section). Both
controls REUSE cells already computed by T3.3's own construction (the design doc's own framing:
"evaluate IT explicitly as the negative control" -- not a new experiment):
  NC1 = the covariate cells' HIGH severity result (large marginal shift, P(Y|X) untouched by
        construction) -- expected: little discrimination/calibration degradation.
  NC2 = the prior cells' LOW severity result (small |Delta-pi|, label prior deliberately altered)
        -- expected: operating-point performance (macro-F1) still moves measurably.
Also restates S1's OWN within-T3.2 negative control (matched row-count single-family subsample,
no mechanism-coverage reduction) for a single combined report, per the user's Phase 3F ask.

Thresholds for "little degradation" / "moves measurably" are fixed here at 0.02 -- the SAME
tolerance already used throughout this Phase 3 codebase (synthesis/phase3_mechanism_lib.py's
AUC_DC_TOL), not a new number invented for this report.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402

DEGRADATION_TOL = 0.02


def _load(name):
    fp = P["out"] / name
    return json.loads(fp.read_text()) if fp.exists() else None


def _nc1_row(domain, cell):
    if cell is None or cell.get("status") != "OK":
        return {"domain": domain, "control": "NC1_large_covariate_PYX_preserved",
               "status": "UNAVAILABLE", "reason": cell.get("status") if cell else "no T3.3 output"}
    hi = cell["severities"].get("high", {})
    little_degradation = (abs(hi.get("delta_auc", 1)) < DEGRADATION_TOL
                          and abs(hi.get("delta_ece", 1)) < DEGRADATION_TOL)
    return {"domain": domain, "control": "NC1_large_covariate_PYX_preserved",
           "status": "OK", "achieved_auc_dc": hi.get("achieved_auc_dc"),
           "delta_auc": hi.get("delta_auc"), "delta_ece": hi.get("delta_ece"),
           "delta_macro_f1": hi.get("delta_macro_f1"), "delta_brier": hi.get("delta_brier"),
           "expected_direction": "little discrimination/calibration degradation even at High severity",
           "expected_direction_held": bool(little_degradation),
           "tolerance": DEGRADATION_TOL}


def _nc2_row(domain, cell):
    if cell is None or cell.get("status") != "OK":
        return {"domain": domain, "control": "NC2_small_prior_PYX_altered",
               "status": "UNAVAILABLE", "reason": cell.get("status") if cell else "no T3.3 output"}
    candidates = {k: v for k, v in cell["severities"].items() if k.startswith("low_") and v.get("feasible")}
    if not candidates:
        return {"domain": domain, "control": "NC2_small_prior_PYX_altered", "status": "INFEASIBLE",
               "reason": "no feasible low-severity direction (see phase3_T33_*.json 'low_up'/'low_down')"}
    key, low = max(candidates.items(), key=lambda kv: abs(kv[1]["delta_macro_f1"]))
    moves_measurably = abs(low.get("delta_macro_f1", 0)) > DEGRADATION_TOL
    return {"domain": domain, "control": "NC2_small_prior_PYX_altered", "status": "OK",
           "direction_used": key, "achieved_delta_pi": low.get("achieved_delta_pi"),
           "delta_auc": low.get("delta_auc"), "delta_macro_f1": low.get("delta_macro_f1"),
           "delta_ece": low.get("delta_ece"),
           "expected_direction": "operating-point performance (macro-F1) moves measurably even at Low severity",
           "expected_direction_held": bool(moves_measurably), "tolerance": DEGRADATION_TOL}


def _s1_own_negative_control(s1: dict) -> list[dict]:
    rows = []
    if not s1:
        return rows
    for level in ("low", "medium", "high"):
        nc_seeds = s1.get("negative_controls", {}).get(level, {}).get("per_seed", [])
        ho_seeds = s1.get("severity_levels", {}).get(level, {}).get("per_seed", [])
        if not nc_seeds or not ho_seeds:
            continue
        nc_recall = [p["target_cicids2017"]["overall_attack_recall"]["recall"] for p in nc_seeds
                    if p["target_cicids2017"]["overall_attack_recall"]["recall"] is not None]
        ho_recall = [p["target_cicids2017"]["overall_attack_recall"]["recall"] for p in ho_seeds
                    if p["target_cicids2017"]["overall_attack_recall"]["recall"] is not None]
        if not nc_recall or not ho_recall:
            continue
        nc_mean = sum(nc_recall) / len(nc_recall); ho_mean = sum(ho_recall) / len(ho_recall)
        rows.append({"domain": "security", "control": f"S1_own_negative_control_{level}",
                    "status": "OK", "negative_control_target_recall_mean": nc_mean,
                    "mechanism_holdout_target_recall_mean": ho_mean,
                    "expected_direction": "negative control degrades LESS than the matched mechanism-holdout condition",
                    "expected_direction_held": bool(nc_mean >= ho_mean - DEGRADATION_TOL)})
    return rows


def run():
    lending = _load("phase3_T33_lending.json")
    security = _load("phase3_T33_security.json")
    s1 = _load("phase3_T32_S1_full.json")

    rows = []
    if lending:
        rows.append(_nc1_row("lending", lending["cells"].get("covariate")))
        rows.append(_nc2_row("lending", lending["cells"].get("prior")))
    if security:
        rows.append(_nc1_row("security", security["cells"].get("covariate")))
        rows.append(_nc2_row("security", security["cells"].get("prior")))
    rows.extend(_s1_own_negative_control(s1))

    import pandas as pd
    pd.DataFrame(rows).to_csv(P["out"] / "negative_controls.csv", index=False)

    all_held = [r["expected_direction_held"] for r in rows if r.get("status") == "OK"]
    report = {
        "rows": rows,
        "n_controls_evaluated": len(all_held),
        "n_expected_direction_held": sum(all_held),
        "summary": ("all evaluated negative controls showed the expected direction" if all_held and all(all_held)
                   else f"{sum(all_held)}/{len(all_held)} evaluated negative controls showed the "
                        "expected direction -- see 'expected_direction_held' per row for which did not"
                   if all_held else "no negative controls were evaluable (see 'status' per row)"),
    }
    (P["out"] / "negative_controls_report.json").write_text(json.dumps(report, indent=2))
    print(f"negative controls: {report['summary']}")
    print("wrote negative_controls.csv, negative_controls_report.json")
    return report


if __name__ == "__main__":
    run()
