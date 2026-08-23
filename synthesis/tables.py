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
from audit.primary_model import all_primary_models  # noqa: E402

DOMAINS = ["clinical", "nlp", "lending", "security"]
TDIR = P["out"] / "tables"
TDIR.mkdir(exist_ok=True)


def _load(kind, domain):
    fp = P["out"] / f"{kind}_{domain}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def _r(x, n=4):
    return None if x is None else round(x, n)


def _primary_seed(seeds):
    return seeds.get("42") or next(iter(seeds.values()))


def t2_master() -> pd.DataFrame:
    """One row per (domain, model, target): every axis TrustShift measures, for EVERY model in
    every domain -- this is the complete supplementary/sensitivity table (PLAN_ijdsa.md T1.8,
    "audit the entire manuscript for any claim that uses a non-primary model ... move all other
    model results into a complete supplementary table"). `is_primary` flags the one row per
    (domain, target) selected by the source-only rule in audit/primary_model.py; the manuscript's
    main-text tables/prose quote ONLY is_primary==True rows. AUC is ranking discrimination;
    macro_f1 is threshold-0.5 operating-point performance, reported alongside it, never merged
    into a single "axis" (PLAN_ijdsa.md T1.1) -- the two are expected to sometimes disagree.
    """
    primary = all_primary_models()
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
            src_dec = src.get("brier_decomposition", {})
            for split, d in s["deltas"].items():
                tgt = s["splits"][split]
                tgt_dec = tgt.get("brier_decomposition", {})
                f1p_src, f1p_tgt = src.get("f1_positive"), tgt.get("f1_positive")
                rows.append({
                    "domain": domain, "model": model, "target": split,
                    "is_primary": model in primary.get(domain, []),
                    "auc_src": round(src["auc"], 4), "auc_tgt": round(tgt["auc"], 4),
                    "delta_auc": round(d["delta_auc"], 4),
                    "macro_f1_src": round(src["macro_f1"], 4), "macro_f1_tgt": round(tgt["macro_f1"], 4),
                    "delta_macro_f1": round(src["macro_f1"] - tgt["macro_f1"], 4),
                    "f1_positive_src": _r(f1p_src), "f1_positive_tgt": _r(f1p_tgt),
                    "delta_f1_positive": _r(f1p_src - f1p_tgt) if (f1p_src is not None and f1p_tgt is not None) else None,
                    "ece_src": round(src["ece"], 4), "ece_tgt": round(tgt["ece"], 4),
                    "delta_ece": round(tgt["ece"] - src["ece"], 4),
                    "brier_src": round(src["brier"], 4), "brier_tgt": round(tgt["brier"], 4),
                    "delta_brier": round(tgt["brier"] - src["brier"], 4),
                    "uncertainty_src": round(src_dec.get("uncertainty", float("nan")), 4),
                    "uncertainty_tgt": round(tgt_dec.get("uncertainty", float("nan")), 4),
                    "delta_uncertainty": round(tgt_dec.get("uncertainty", float("nan")) - src_dec.get("uncertainty", float("nan")), 4),
                    "reliability_src": round(src_dec.get("reliability", float("nan")), 4),
                    "reliability_tgt": round(tgt_dec.get("reliability", float("nan")), 4),
                    "resolution_src": round(src_dec.get("resolution", float("nan")), 4),
                    "resolution_tgt": round(tgt_dec.get("resolution", float("nan")), 4),
                    "delta_gap_auc": None if d.get("delta_gap_auc") is None else round(d["delta_gap_auc"], 4),
                    "delta_gap_ci": d.get("delta_gap_ci"),
                    "delta_gap_p": d.get("delta_gap_p"),
                    "diagnosis": summ.get(split, {}).get("diagnosis"),
                    "diagnosis_note": summ.get(split, {}).get("note"),
                    "auc_dc": summ.get(split, {}).get("auc_dc"),
                })
    return pd.DataFrame(rows)


def t2_primary() -> pd.DataFrame:
    """The main-text subset of T2_master: exactly one model per (domain, target), selected by
    the source-only rule (audit/primary_model.py), never by which target result reads best."""
    full = t2_master()
    return full[full["is_primary"]].reset_index(drop=True)


def _load_remediation_files(domain: str) -> list[dict]:
    """Single-experiment domains write remediation_{domain}.json; multi-experiment domains
    (lending: temporal + geographic are different experiments -- see audit/primary_model.py)
    write one remediation_{domain}_{model}.json per experiment."""
    single = _load("remediation", domain)
    if single is not None:
        return [single]
    return [json.loads(fp.read_text())
            for fp in sorted(P["out"].glob(f"remediation_{domain}_*.json"))]


def t5_remediation() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        for r in _load_remediation_files(domain):
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
                         "note": sm.get("note"),
                         "max_delta_pi": round(sm["max_delta_pi"], 4), "auc_dc": sm["auc_dc"],
                         "auc_source": con.get("auc_source"), "auc_source_reweighted": con.get("auc_source_reweighted"),
                         "auc_target": con.get("auc_target"),
                         "concept_adjudicable": con.get("adjudicable")})
    return pd.DataFrame(rows)


def t_probe_availability() -> pd.DataFrame:
    """Which Stage-A/Stage-B probes actually produced evidence, per domain (PLAN_ijdsa.md T1.6:
    "add the probe-availability matrix as a real table" -- BLOCKER-4)."""
    rows = []
    for domain in DOMAINS:
        d = _load("diagnosis", domain)
        if not d:
            continue
        avail = d.get("probe_availability", {})
        rows.append({
            "domain": domain,
            "label_shift_probe": avail.get("label_shift", True),
            "covariate_probe": avail.get("covariate", bool(d.get("covariate"))),
            "concept_probe": avail.get("concept", bool(d.get("concept"))),
        })
    return pd.DataFrame(rows)


def main():
    t2 = t2_master(); t2.to_csv(TDIR / "T2_master.csv", index=False)
    t2p = t2_primary(); t2p.to_csv(TDIR / "T2_primary.csv", index=False)
    t4 = t4_diagnosis(); t4.to_csv(TDIR / "T4_diagnosis.csv", index=False)
    t5 = t5_remediation(); t5.to_csv(TDIR / "T5_remediation.csv", index=False)
    tprobe = t_probe_availability(); tprobe.to_csv(TDIR / "T_probe_availability.csv", index=False)
    meta = _load("meta", "analysis") if (P["out"] / "meta_analysis.json").exists() else None
    print("T2 master, ALL models (complete supplementary/sensitivity table):")
    print(t2.to_string(index=False) if not t2.empty else "  (empty)")
    print("\nT2 primary (source-only rule, one row per domain x target -- what the main text quotes):")
    print(t2p.to_string(index=False) if not t2p.empty else "  (empty)")
    print("\nProbe availability:")
    print(tprobe.to_string(index=False) if not tprobe.empty else "  (empty)")
    print(f"\nwrote T2_master({len(t2)}), T2_primary({len(t2p)}), T4({len(t4)}), T5({len(t5)}), "
          f"T_probe_availability({len(tprobe)}) to {TDIR}")


if __name__ == "__main__":
    main()
