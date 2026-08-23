"""Round-5 scientific-audit-gate regression guards (item 8, item 14).

item 8: S1's cross-dataset target-recall result was reclassified non-informative for H2 (round-4,
item I) after it was found to be flat regardless of severity or negative control. This must stay
excluded from H2's vote permanently -- these tests fail loudly if a future edit accidentally lets
it back in, rather than silently inflating confidence on a false premise.

item 14: H3 was redefined and downgraded from "mixed" to "not_formally_tested" because the one
analysis that would formally test it (magnitude vs. mechanism, leave-one-out cross-validated)
returns negative out-of-sample R^2 for every candidate model. This must not silently revert to
claiming a formal result without a formal comparison that actually works.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402


def _run_gate():
    from synthesis.phase3_hypothesis_gate import run
    return run()


def test_s1_cross_dataset_excluded_from_h2_vote():
    gate = _run_gate()
    h2 = gate["H2_magnitude_insufficient_within_mechanism"]
    s1_cross = h2["evidence_hierarchy"]["level_3_supporting_stress_test"].get("S1_cross_dataset_target_recall")
    assert s1_cross is not None, "S1 cross-dataset entry missing entirely -- re-run T3.2/S1 and phase3_global_synthesis first"
    assert s1_cross["status"] == "non_informative"
    assert s1_cross["weight"].startswith("EXCLUDED")
    # the decisive H2 status must be traceable to Level 1 language in its own reason string, and
    # must NOT cite S1/level-3 as the deciding evidence
    assert "Level 1" in h2["reason"]
    assert "S1" not in h2["reason"].split("Level 3")[0].split("Level 1")[1].split(".")[0], (
        "H2's stated reason should not attribute its Level-1 verdict to S1 -- S1 belongs only in "
        "the Level 3 (non-decisive) discussion"
    )


def test_h2_cites_context_vs_magnitude_headtohead_not_just_low_r2():
    """Item 4/5: H2 must not rest on 'magnitude's R2~=0' in isolation -- it must cite a real
    head-to-head comparison (state identity vs. magnitude) as the round-5 audit added."""
    gate = _run_gate()
    h2 = gate["H2_magnitude_insufficient_within_mechanism"]
    ctx = h2["evidence_hierarchy"]["level_1_naturally_occurring"].get("T3.1_state_identity_vs_magnitude_headtohead")
    assert ctx is not None, "context-vs-magnitude head-to-head evidence missing from H2 -- run synthesis/phase3_h2_scientific_audit.py first"
    assert "scope_caveat" in ctx and "MECHANISM" in ctx["scope_caveat"], (
        "the context-vs-magnitude finding must explicitly disclaim that it is NOT mechanism "
        "evidence -- omitting this caveat would repeat the exact overclaim the user flagged"
    )


def test_h3_not_claimed_mixed_when_formal_test_is_underpowered():
    gate = _run_gate()
    h3 = gate["H3_staged_audit_beats_scalar"]
    mvm_fp = P["out"] / "phase3_h2_magnitude_vs_mechanism.json"
    if not mvm_fp.exists():
        pytest.skip("phase3_h2_magnitude_vs_mechanism.json not present -- run synthesis/phase3_h2_scientific_audit.py first")
    mvm = json.loads(mvm_fp.read_text())
    all_loo_negative = all(
        (mvm.get(m, {}).get("loo_cv_r2") or -1) <= 0
        for m in ("M0_magnitude_only", "M1_mechanism_only", "M2_magnitude_plus_mechanism")
    )
    if all_loo_negative:
        assert h3["status"] == "not_formally_tested", (
            "H3 must report 'not_formally_tested' (not 'mixed') when its formal comparison's "
            "out-of-sample R2 is negative for every candidate model -- claiming 'mixed' here would "
            "assert a formal result without a formal comparison that actually works (item 14)"
        )
    assert "formal_test_attempted" in h3 and h3["formal_test_attempted"]


def test_h2a_and_h2b_are_separate_top_level_verdicts_never_conflated():
    """Round-6: 'state identity beats magnitude' (H2a) and 'mechanism beats magnitude' (H2b) are
    different hypotheses. This guards against a future edit silently collapsing them back into one
    'H2: supported' headline, or letting H2a's strong evidence leak into H2b's status."""
    gate = _run_gate()
    assert "H2a_scalar_magnitude_insufficient" in gate
    assert "H2b_mechanism_beyond_magnitude" in gate
    h2a, h2b = gate["H2a_scalar_magnitude_insufficient"], gate["H2b_mechanism_beyond_magnitude"]
    assert "mechanism" not in h2a["claim"].lower(), "H2a's claim text must not mention mechanism"
    assert "magnitude" in h2b["claim"].lower() and "mechanism" in h2b["claim"].lower()
    assert h2b["status"] == "not_established", (
        "H2b must not be silently upgraded to 'supported' while the direct mechanism-vs-magnitude "
        "test (results/mechanism_vs_magnitude_final.csv) has negative out-of-sample R2 for every model"
    )


def test_no_downstream_script_hardcodes_s1_cross_dataset_as_h2_support():
    """Source-grep regression guard: the exact bug pattern (S1's cross-dataset recall counted as
    confirmatory H2 evidence) must not reappear in the synthesis layer."""
    src = (Path(__file__).resolve().parents[1] / "synthesis" / "phase3_hypothesis_gate.py").read_text()
    assert 'S1_cross_dataset_target_recall' in src
    # the weight string for this entry must always say EXCLUDED in the source itself, not just at runtime
    idx = src.index("S1_cross_dataset_target_recall")
    nearby = src[idx: idx + 800]
    assert "EXCLUDED" in nearby, (
        "S1_cross_dataset_target_recall's weight must say EXCLUDED near its definition in source"
    )
