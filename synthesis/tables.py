"""Paper tables (PLAN.md Task 9). Reads results/*.json -> results/tables/*.csv.

T1 dataset/shift-pair summary   T2 master axes x domains grid (the leaderboard)
T3 per-subgroup dAUC (who pays)  T4 diagnosis triples   T5 remediation grid
Every number here traces to an audit/diagnosis/remediation/meta JSON; nothing hardcoded.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402

DOMAINS = ["clinical", "nlp", "lending", "security"]
TDIR = P["out"] / "tables"
TDIR.mkdir(exist_ok=True)


def _load(kind, domain):
    fp = P["out"] / f"{kind}_{domain}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def _primary_seed(seeds):
    return seeds.get("42") or next(iter(seeds.values()))


def t2_master() -> pd.DataFrame:
    """One row per (domain, model, target): AUC/ECE source->target, dAUC, dG + CI, diagnosis."""
    rows = []
    for domain in DOMAINS:
        a = _load("audit", domain)
        if not a:
            continue
        diag = _load("diagnosis", domain) or {}
        summ = diag.get("summary", {})
        for model, seeds in a["models"].items():
            s = _primary_seed(seeds)
            src = s["splits"]["source_test"]
            for split, d in s["deltas"].items():
                tgt = s["splits"][split]
                rows.append({
                    "domain": domain, "model": model, "target": split,
                    "auc_src": round(src["auc"], 4), "auc_tgt": round(tgt["auc"], 4),
                    "delta_auc": round(d["delta_auc"], 4),
                    "ece_src": round(src["ece"], 4), "ece_tgt": round(tgt["ece"], 4),
                    "delta_ece": round(tgt["ece"] - src["ece"], 4),
                    "delta_gap_auc": None if d.get("delta_gap_auc") is None else round(d["delta_gap_auc"], 4),
                    "delta_gap_ci": d.get("delta_gap_ci"),
                    "delta_gap_p": d.get("delta_gap_p"),
                    "diagnosis": summ.get(split, {}).get("diagnosis"),
                    "auc_dc": summ.get(split, {}).get("auc_dc"),
                })
    return pd.DataFrame(rows)


def t5_remediation() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        r = _load("remediation", domain)
        if not r:
            continue
        for tgt, m in r["targets"].items():
            rows.append({"domain": domain, "model": r["model"], "target": tgt,
                         "ece_L0": round(m["ece_L0"], 4), "ece_isotonic": round(m["ece_isotonic"], 4),
                         "ece_temperature": round(m["ece_temperature"], 4),
                         "auc_L0": round(m["auc_L0"], 4), "auc_isotonic": round(m["auc_isotonic"], 4),
                         "gap_L0": round(m["gap_L0"], 4) if m["gap_L0"] == m["gap_L0"] else None,
                         "gap_isotonic": round(m["gap_isotonic"], 4) if m["gap_isotonic"] == m["gap_isotonic"] else None})
    return pd.DataFrame(rows)


def t4_diagnosis() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        d = _load("diagnosis", domain)
        if not d:
            continue
        for tgt, sm in d.get("summary", {}).items():
            con = d.get("concept", {}).get(tgt, {})
            rows.append({"domain": domain, "target": tgt, "diagnosis": sm["diagnosis"],
                         "max_delta_pi": round(sm["max_delta_pi"], 4), "auc_dc": sm["auc_dc"],
                         "auc_source": con.get("auc_source"), "auc_source_reweighted": con.get("auc_source_reweighted"),
                         "auc_target": con.get("auc_target")})
    return pd.DataFrame(rows)


def main():
    t2 = t2_master(); t2.to_csv(TDIR / "T2_master.csv", index=False)
    t4 = t4_diagnosis(); t4.to_csv(TDIR / "T4_diagnosis.csv", index=False)
    t5 = t5_remediation(); t5.to_csv(TDIR / "T5_remediation.csv", index=False)
    meta = _load("meta", "analysis") if (P["out"] / "meta_analysis.json").exists() else None
    print("T2 master (axes x domains):")
    print(t2.to_string(index=False) if not t2.empty else "  (empty)")
    print(f"\nwrote T2({len(t2)}), T4({len(t4)}), T5({len(t5)}) to {TDIR}")


if __name__ == "__main__":
    main()
