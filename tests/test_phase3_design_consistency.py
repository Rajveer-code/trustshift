"""Pre-Phase-3 consistency guard (user directive, 2026-08-23 round-3 review).

Checks that docs/PHASE3_DESIGN.md (the frozen design) and the actual codebase have not drifted
apart -- run this before, and again after, any Phase 3 code is written. It does not validate the
SCIENCE of the design (that was the point of the human review rounds); it validates that the
design's stated commitments are still findable in both the doc and the code that will implement
them, so an implementer cannot silently drop a gate the design requires.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import P  # noqa: E402

DESIGN = Path(__file__).resolve().parents[1] / "docs" / "PHASE3_DESIGN.md"


def _design_text() -> str:
    if not DESIGN.exists():
        pytest.skip("docs/PHASE3_DESIGN.md not present (gitignored -- run from a checkout that has it)")
    return DESIGN.read_text(encoding="utf-8")


def test_primary_model_mapping_exists_for_every_domain():
    fp = P["out"] / "primary_models.json"
    assert fp.exists(), "run audit/primary_model.py before Phase 3"
    sel = json.loads(fp.read_text())
    for domain in ("clinical", "nlp", "lending", "security"):
        assert domain in sel, f"no primary model selected for {domain}"
        assert len(sel[domain]) >= 1
    # lending is the one domain with two non-comparable experiments (temporal + geographic) --
    # the design doc and audit/primary_model.py must agree it is NOT collapsed to one model.
    assert len(sel["lending"]) == 2, (
        "lending must have exactly two primary models (temporal + geographic experiments are "
        "not competing models for one task -- see audit/primary_model.py docstring); if this "
        "changes, PHASE3_DESIGN.md's T3.1 section must be updated to match"
    )


def test_design_severity_labels_present():
    text = _design_text()
    for label in ("Low severity", "Medium severity", "High severity"):
        assert label in text, f"design doc missing severity level: {label}"


def test_design_exclusion_thresholds_present():
    text = _design_text()
    assert "500 total rows" in text, "T3.1 minimum total-N gate not found in design doc"
    assert re.search(r"30 rows", text), "T3.1 minimum subgroup-N gate not found in design doc"
    assert re.search(r"10 positive", text), "T3.1 minimum positive/negative-count gate not found"


def test_design_falsification_criteria_present():
    text = _design_text()
    for h in ("H1", "H2", "H3"):
        assert re.search(rf"\*\*{h}\b", text), f"{h} not defined with the expected heading format"
    for verdict in ("Supported", "Mixed", "Falsified", "*Supported*", "*Mixed*", "*Falsified*"):
        pass  # exact casing varies (bold vs italic) -- checked more precisely below
    lowered = text.lower()
    for verdict in ("supported", "mixed", "falsified"):
        assert verdict in lowered, f"falsification decision rule missing verdict category: {verdict}"


def test_design_primary_endpoints_named():
    text = _design_text()
    assert "Primary endpoints" in text
    assert "Secondary endpoints" in text
    # the design's stated primary endpoint style must match what audit/engine.py actually
    # computes (AUC_dc as a continuous covariate, not only tercile bins)
    assert "continuous" in text.lower(), (
        "design doc must state the T3.1 primary endpoint is against continuous AUC_dc, not "
        "tercile-binned (round-2 review requirement)"
    )


def test_design_does_not_reintroduce_forbidden_terms_as_active_claims():
    """The design doc MAY discuss why these terms are wrong (that's the whole point of several
    sections), but must never assert them as the paper's own diagnosis. This checks for the
    specific FALSE-CLAIM patterns, not the terms in isolation (which appear legitimately in
    explanatory prose throughout the doc)."""
    text = _design_text()
    forbidden_patterns = [
        r"is\s+label\s+shift\b(?!.{0,80}not)",  # "is label shift" without a nearby "not"
        r"the mechanism is label shift",
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, text, re.IGNORECASE), (
            f"design doc appears to assert a forbidden claim matching: {pat!r}"
        )
    # the locked terminology must be present verbatim (this is the phrase the user fixed)
    assert "novel attack-mechanism" in text.lower() or "unseen attack-family" in text.lower()


def test_security_s2_verdict_is_recorded():
    text = _design_text()
    assert "NOT APPLICABLE" in text or "not applicable" in text.lower(), (
        "S2 feasibility verdict must be recorded in the design doc before Phase 3 runs"
    )


def test_t33_dataset_decision_is_recorded():
    text = _design_text()
    assert "HMDA+Security" in text or "Lending + Security" in text or "Lending+Security" in text
    assert "HMDA+NLP" in text or "Lending + NLP" in text or "Lending+NLP" in text


def test_no_downstream_module_hardcodes_a_non_primary_model():
    """Regression guard for the two bugs the primary-model consistency audit found
    (audit/diagnosis.py's NLP concept probe hardcoded to 'bert'; clinical KS-proxy with no
    model filter at all). If either pattern reappears, this test fails before it reaches the
    manuscript."""
    diagnosis_src = (Path(__file__).resolve().parents[1] / "audit" / "diagnosis.py").read_text()
    assert 'model == "bert"' not in diagnosis_src, (
        "audit/diagnosis.py hardcodes model=='bert' again -- use select_primary_models('nlp')[0]"
    )
    # the clinical branch must filter by model before computing the KS-distance proxy
    assert "df42.model == ks_model" in diagnosis_src or "model == ks_model" in diagnosis_src, (
        "clinical KS-distance proxy must filter to the primary model (previous bug: no filter "
        "at all, pooling every model's rows together)"
    )
