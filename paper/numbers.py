"""Single source of every quantitative claim in the manuscript.

Pulls values straight from results/*.json and results/tables/*.csv into paper/numbers.json.
The manuscript must cite only keys that appear here. `--check` greps main.tex for hardcoded
numbers that drift from these (best-effort). Run after any results change.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402


def collect() -> dict:
    t2 = pd.read_csv(P["out"] / "tables" / "T2_master.csv")
    t5 = pd.read_csv(P["out"] / "tables" / "T5_remediation.csv")
    meta = json.loads((P["out"] / "meta_analysis.json").read_text())
    rec = json.loads((P["out"] / "security_family_recall.json").read_text())

    def row(dom, model, tgt):
        r = t2[(t2.domain == dom) & (t2.model == model) & (t2.target == tgt)].iloc[0]
        return r

    n = {}
    # --- clinical ---
    c = row("clinical", "fedavg", "target_brfss")
    n["clinical_fedavg"] = dict(auc_src=c.auc_src, auc_tgt=c.auc_tgt, dauc=c.delta_auc,
                                ece_src=c.ece_src, ece_tgt=c.ece_tgt, dgap=c.delta_gap_auc,
                                dgap_ci=json.loads(c.delta_gap_ci) if isinstance(c.delta_gap_ci, str) else c.delta_gap_ci,
                                dgap_p=c.delta_gap_p)
    cx = row("clinical", "xgb", "target_brfss")
    n["clinical_xgb"] = dict(ece_tgt=cx.ece_tgt, dgap=cx.delta_gap_auc)
    # --- nlp (range across 4 models x 2 targets) ---
    nlp = t2[t2.domain == "nlp"]
    n["nlp"] = dict(auc_src_min=nlp.auc_src.min(), auc_src_max=nlp.auc_src.max(),
                    auc_tgt_min=nlp.auc_tgt.min(), auc_tgt_max=nlp.auc_tgt.max(),
                    dauc_min=nlp.delta_auc.min(), dauc_max=nlp.delta_auc.max(),
                    dgap_min=nlp.delta_gap_auc.min(), dgap_max=nlp.delta_gap_auc.max(),
                    ece_tgt_min=nlp.ece_tgt.min(), ece_tgt_max=nlp.ece_tgt.max())
    # --- lending temporal ---
    lt = t2[(t2.domain == "lending") & (t2.model == "lightgbm_temporal")]
    n["lending_temporal"] = dict(dauc_min=lt.delta_auc.min(), dauc_max=lt.delta_auc.max(),
                                 dgap_min=lt.delta_gap_auc.min(), dgap_max=lt.delta_gap_auc.max(),
                                 auc_dc_min=lt.auc_dc.min(), auc_dc_max=lt.auc_dc.max())
    lg = row("lending", "lightgbm_geo", "target_heldout_states")
    n["lending_geo"] = dict(dauc=lg.delta_auc, dgap=lg.delta_gap_auc)
    # --- security ---
    s = row("security", "lightgbm", "target_cicids2017")
    n["security"] = dict(auc_src=s.auc_src, auc_tgt=s.auc_tgt, dauc=s.delta_auc,
                         ece_src=s.ece_src, ece_tgt=s.ece_tgt)
    tr = rec["target_cicids2017"]
    n["security_recall"] = {k: round(v, 3) for k, v in tr.items()}
    # --- remediation (isotonic) ---
    n["remediation"] = {}
    for _, r in t5.iterrows():
        n["remediation"][f"{r.domain}/{r.target}"] = dict(
            ece_L0=r.ece_L0, ece_iso=r.ece_isotonic, auc_L0=r.auc_L0, auc_iso=r.auc_isotonic)
    # --- meta ---
    n["meta"] = {k: dict(slope=round(meta[k]["slope"], 3), ci=[round(x, 3) for x in meta[k]["ci"]],
                         r2=round(meta[k]["r2"], 3), n=meta[k]["n"])
                 for k in ("M1_gap_vs_calibration", "M2_gap_vs_covariateshift", "M3_gap_vs_accuracyloss")}
    n["meta"]["n_shift_points"] = meta["n_shift_points"]
    return n


def main():
    n = collect()
    (Path(__file__).parent / "numbers.json").write_text(json.dumps(n, indent=2))
    print(json.dumps(n, indent=2))


if __name__ == "__main__":
    main()
