"""Primary-model consistency audit across the full pipeline (user directive, 2026-08-22,
batch item 12/D): audit -> diagnosis -> remediation -> meta-analysis -> tables -> figures ->
manuscript must never silently reference a different model than audit/primary_model.py selects.

This script does not assume the pipeline is consistent -- it inspects each stage's actual
model-selection logic and reports mismatches. It already caught two real bugs before this file
existed: audit/diagnosis.py's NLP concept probe was hardcoded to "bert" while the source-only rule
selects "roberta", and the clinical KS-distance covariate proxy had no model filter at all,
silently pooling both models' scores. Both are fixed (see audit/diagnosis.py); this script is the
regression guard against a third one appearing.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402
from audit.primary_model import all_primary_models  # noqa: E402


def _remediation_models(domain: str) -> list[str]:
    single = P["out"] / f"remediation_{domain}.json"
    if single.exists():
        return [json.loads(single.read_text())["model"]]
    return [json.loads(fp.read_text())["model"]
            for fp in sorted(P["out"].glob(f"remediation_{domain}_*.json"))]


def _diagnosis_concept_model(domain: str) -> str | None:
    fp = P["out"] / f"diagnosis_{domain}.json"
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())
    return d.get("concept_probe_model") or d.get("p_hat_ks_model")


def _table_models(domain: str) -> list[str]:
    fp = P["out"] / "tables" / "T2_primary.csv"
    if not fp.exists():
        return []
    import pandas as pd
    t2 = pd.read_csv(fp)
    return sorted(t2.loc[t2.domain == domain, "model"].unique().tolist())


# The manuscript uses display names / experiment labels, not the pipeline's internal model
# keys (e.g. "FedAvg" not "fedavg", "temporal" not "lightgbm_temporal"). This maps every
# manuscript surface form to its canonical internal key so the comparison checks the same
# underlying model identity, not string formatting.
_MANUSCRIPT_ALIAS = {
    "fedavg": "fedavg", "xgb": "xgb", "roberta": "roberta", "bert": "bert",
    "mentalbert": "mentalbert", "mentalroberta": "mentalroberta",
    "xgboost": "xgboost", "lightgbm": "lightgbm",
    "temporal": "lightgbm_temporal", "geographic": "lightgbm_geo",
    "lightgbm_temporal": "lightgbm_temporal", "lightgbm_geo": "lightgbm_geo",
}


def _manuscript_models(domain: str, tex_path: Path) -> list[str]:
    """Best-effort scrape: which model names for this domain appear in main.tex near the
    domain's Table 3 row / prose, normalized to internal model keys via _MANUSCRIPT_ALIAS.
    Manual verification is still required; this is a smoke check, not a proof."""
    text = tex_path.read_text(encoding="utf-8")
    tag = {"clinical": "Clinical (", "nlp": "NLP (", "lending": "Lending (",
           "security": "Security ("}.get(domain)
    if tag is None:
        return []
    found = re.findall(rf"{re.escape(tag)}([A-Za-z0-9_]+)\)", text)
    normalized = {_MANUSCRIPT_ALIAS.get(f.lower(), f.lower()) for f in found}
    return sorted(normalized)


def run() -> dict:
    primary = all_primary_models()
    tex_path = P["out"].parent / "paper" / "main.tex"
    report = {}
    for domain, primary_models in primary.items():
        audit_d = json.loads((P["out"] / f"audit_{domain}.json").read_text())
        audit_models = sorted(audit_d["models"].keys())  # audit computes for EVERY model --
        # "consistent" here means primary_models subset audit_models, not equality.
        diag_model = _diagnosis_concept_model(domain)
        rem_models = _remediation_models(domain)
        tab_models = _table_models(domain)
        ms_models = _manuscript_models(domain, tex_path)

        row = {
            "selected_primary_models": primary_models,
            "audit_models_present": audit_models,
            "audit_consistent": all(m in audit_models for m in primary_models),
            "diagnosis_concept_or_proxy_model": diag_model,
            "diagnosis_consistent": (diag_model in primary_models) if diag_model else None,
            "remediation_models": rem_models,
            "remediation_consistent": sorted(rem_models) == sorted(primary_models),
            "table_primary_models": tab_models,
            "table_consistent": sorted(tab_models) == sorted(primary_models),
            "manuscript_models_found": ms_models,
            "manuscript_consistent": (sorted(ms_models) == sorted(primary_models)
                                      if ms_models else None),
        }
        row["explicit_note"] = _explain(domain, row)
        report[domain] = row
    return report


def _explain(domain: str, row: dict) -> str:
    if domain == "clinical":
        return ("diagnosis_concept_or_proxy_model refers to the KS-distance SCORE proxy (no "
                "AUC_dc probe exists for this domain); it is fixed to the source-only primary "
                "model (fedavg) as of the 2026-08-22 consistency fix.")
    if domain == "nlp":
        return ("diagnosis_concept_or_proxy_model is the concept-shift reweighting probe's "
                "model, fixed to the source-only primary model (roberta) as of the 2026-08-22 "
                "consistency fix; the covariate probe (AUC_dc) is model-independent (built from "
                "raw text features, not any model's predictions), so it has no model identity "
                "to check.")
    if domain == "lending":
        return ("lending has TWO primary models (temporal + geographic are different "
                "experiments, not competing models -- audit/primary_model.py docstring). No "
                "concept probe exists for lending (feature<->prediction alignment unavailable), "
                "so diagnosis_concept_or_proxy_model is None by design, not a gap.")
    if domain == "security":
        return ("security has no covariate or concept probe (probe_availability: both False); "
                "diagnosis_concept_or_proxy_model is None by design. manuscript_models_found "
                "should show only xgboost -- lightgbm's numbers are reported in the supplementary "
                "T2_master.csv sensitivity table, not the main-text prose/tables.")
    return ""


def main():
    report = run()
    (P["out"] / "primary_model_consistency_report.json").write_text(
        json.dumps(report, indent=2))
    all_ok = True
    for domain, row in report.items():
        flags = [k for k in ("audit_consistent", "diagnosis_consistent", "remediation_consistent",
                             "table_consistent", "manuscript_consistent")
                 if row.get(k) is False]
        status = "OK" if not flags else f"MISMATCH: {flags}"
        if flags:
            all_ok = False
        print(f"  {domain:10s} primary={row['selected_primary_models']}  {status}")
    print(f"\nwrote {P['out'] / 'primary_model_consistency_report.json'}")
    print("ALL CONSISTENT" if all_ok else "INCONSISTENCIES FOUND -- see report")
    return report


if __name__ == "__main__":
    main()
