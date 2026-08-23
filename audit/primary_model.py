"""Source-only primary-model selection rule (PLAN_ijdsa.md T1.8, BLOCKER-5 fix).

Selects, for each domain, the primary model(s) whose numbers are quoted in the manuscript's
abstract, Discussion, and prose ranges (T2_master.csv already reports every model as a complete
sensitivity table). The selection uses ONLY source_test quantities -- no target-split number may
enter this decision, so the choice cannot be steered toward whichever model tells the most
favorable target-side story. That silent steering is exactly what BLOCKER-5 documented in the
original manuscript (clinical: fedavg's narrowing gap quoted, xgb's widening gap omitted;
security: lightgbm's 0.635 quoted, xgboost's 0.706 omitted).

IMPORTANT scoping rule: models are only compared against OTHER MODELS ATTEMPTING THE SAME SHIFT
EXPERIMENT -- i.e. the same set of target splits. lending's lightgbm_temporal (targets:
2022/2023/2024) and lightgbm_geo (target: heldout_states) are two DIFFERENT scientific
experiments (temporal vs geographic shift), not two candidate models for one task; comparing
their source AUCs against each other and discarding one would silently delete an entire
experiment, which is a worse error than the one this rule exists to fix. Models are grouped by
their target-split signature (the frozenset of `deltas` keys) and the rule is applied only WITHIN
each group. A domain can therefore have more than one primary model (lending currently has two:
one per experiment); a domain where every model shares one target-split signature (clinical, nlp,
security) still gets exactly one.

Rule (config.PRIMARY_MODEL_TIEBREAK_PRIORITY documents the domain-specific tiebreak order),
applied within each target-signature group:
  1. Highest source_test AUC.
  2. Ties (rounded to 1e-9) broken by lower source_test ECE.
  3. Remaining ties broken by a FIXED, pre-declared model-name priority list in config.py --
     never chosen post hoc, never adjusted after seeing target results.

Every other model is still computed, reported, and released in full in T2_master.csv -- this
rule decides what appears in the MAIN manuscript table/prose, not what gets computed or disclosed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P, PRIMARY_MODEL_TIEBREAK_PRIORITY  # noqa: E402


def _target_signature(seeds_dict) -> frozenset:
    s = seeds_dict.get("42") or next(iter(seeds_dict.values()))
    return frozenset(s["deltas"].keys())


def select_primary_models(domain: str, audit: dict | None = None) -> list[str]:
    """Returns the list of primary models for `domain` -- one per distinct target-split
    signature group (see module docstring). Length 1 for domains where every model shares the
    same target set; length > 1 for domains (currently lending) with multiple non-comparable
    experiments living under one domain name."""
    if audit is None:
        audit = json.loads((P["out"] / f"audit_{domain}.json").read_text())
    groups: dict[frozenset, list[tuple[str, float, float]]] = {}
    for model, seeds in audit["models"].items():
        s = seeds.get("42") or next(iter(seeds.values()))
        src = s["splits"].get("source_test")
        if src is None or src.get("auc") is None:
            continue
        sig = _target_signature(seeds)
        groups.setdefault(sig, []).append((model, src["auc"], src.get("ece", float("inf"))))
    if not groups:
        raise ValueError(f"no models with a source_test AUC for domain={domain}")

    priority = PRIMARY_MODEL_TIEBREAK_PRIORITY.get(domain, [])

    def rank_key(item):
        model, auc, ece = item
        pr = priority.index(model) if model in priority else len(priority)
        return (-round(auc, 9), round(ece, 9), pr)

    primaries = []
    for sig, candidates in groups.items():
        candidates.sort(key=rank_key)
        primaries.append(candidates[0][0])
    return primaries


def select_primary_model(domain: str, audit: dict | None = None) -> str:
    """Convenience accessor for domains known to have exactly one experiment (raises if not --
    callers that may see multiple primaries, e.g. anything touching lending, must use
    select_primary_models (plural) instead)."""
    primaries = select_primary_models(domain, audit)
    if len(primaries) != 1:
        raise ValueError(
            f"domain={domain} has {len(primaries)} primary models ({primaries}) across "
            "distinct experiments -- use select_primary_models() (plural) and handle each "
            "experiment separately, do not pick just one."
        )
    return primaries[0]


def all_primary_models() -> dict:
    """domain -> list of primary models (see select_primary_models)."""
    out = {}
    for domain in ["clinical", "nlp", "lending", "security"]:
        fp = P["out"] / f"audit_{domain}.json"
        if fp.exists():
            out[domain] = select_primary_models(domain)
    return out


def main():
    sel = all_primary_models()
    (P["out"] / "primary_models.json").write_text(json.dumps(sel, indent=2))
    print("primary model(s) per domain (source-only rule, PLAN_ijdsa.md T1.8):")
    for d, models in sel.items():
        tag = "" if len(models) == 1 else "  <- multiple experiments, see module docstring"
        print(f"  {d:10s} -> {models}{tag}")
    print(f"\nwrote {P['out'] / 'primary_models.json'}")


if __name__ == "__main__":
    main()
