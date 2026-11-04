"""
tests/test_agent_mock.py — Tests for agent guardrails and tools.

Tests guardrails and tool functions as plain Python calls.
Does NOT test the full LangGraph ReAct loop (mocking bind_tools
is fragile and adds no coverage over testing the functions directly).
"""
from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------
# Guardrails tests
# ------------------------------------------------------------------

class TestGuardrails:
    """Tests for agent/guardrails.py validation logic."""

    def test_guardrails_pass_with_valid_input(self):
        """validate_str_draft passes with sufficient evidence."""
        from agent.guardrails import validate_str_draft

        result = validate_str_draft(
            str_text="This account shows suspicious FATF-flagged patterns.",
            tool_call_count=5,
            citations=[
                {"text": "FATF Recommendation 20 requires...", "source": "fatf-2012.pdf", "page": 15},
            ],
            fraud_probability=0.85,
            evidence_threshold=0.6,
        )
        assert result.passed is True
        assert len(result.violations) == 0

    def test_guardrails_fail_no_tools(self):
        """G1: Fails when tool_call_count < 3."""
        from agent.guardrails import validate_str_draft

        result = validate_str_draft(
            str_text="Some STR text",
            tool_call_count=1,
            citations=[{"text": "some reg", "source": "doc.pdf", "page": 1}],
            fraud_probability=0.9,
        )
        assert result.passed is False
        assert any("G1" in v for v in result.violations)

    def test_guardrails_fail_no_citations(self):
        """G2: Fails when citations list is empty."""
        from agent.guardrails import validate_str_draft

        result = validate_str_draft(
            str_text="Some STR text",
            tool_call_count=5,
            citations=[],
            fraud_probability=0.9,
        )
        assert result.passed is False
        assert any("G2" in v for v in result.violations)

    def test_guardrails_fail_no_risk_score(self):
        """G3: Fails when fraud_probability is None."""
        from agent.guardrails import validate_str_draft

        result = validate_str_draft(
            str_text="Some STR text",
            tool_call_count=5,
            citations=[{"text": "some reg", "source": "doc.pdf", "page": 1}],
            fraud_probability=None,
        )
        assert result.passed is False
        assert any("G3" in v for v in result.violations)

    def test_guardrails_warn_low_risk(self):
        """G3: Warns when fraud_probability < evidence_threshold."""
        from agent.guardrails import validate_str_draft

        result = validate_str_draft(
            str_text="Some STR text",
            tool_call_count=5,
            citations=[{"text": "some reg", "source": "doc.pdf", "page": 1}],
            fraud_probability=0.3,
            evidence_threshold=0.6,
        )
        # Should pass (not a violation) but have a warning
        assert result.passed is True
        assert any("G3" in w for w in result.warnings)

    def test_guardrails_hallucination_warning(self):
        """G4: Warns when FATF is mentioned but not in citations."""
        from agent.guardrails import validate_str_draft

        result = validate_str_draft(
            str_text="According to FATF recommendations, this is suspicious.",
            tool_call_count=5,
            citations=[
                {"text": "RBI master direction on KYC", "source": "rbi-kyc.pdf", "page": 5},
            ],
            fraud_probability=0.9,
        )
        # FATF mentioned but not in citations -> G4 warning
        assert any("G4" in w for w in result.warnings)

    def test_guardrails_no_hallucination_for_legislative_refs(self):
        """PMLA, RBI, FIU-IND are whitelisted and should not trigger G4."""
        from agent.guardrails import validate_str_draft

        result = validate_str_draft(
            str_text="Filed under PMLA 2002 with FIU-IND per RBI guidelines.",
            tool_call_count=5,
            citations=[
                {"text": "Some regulatory text", "source": "doc.pdf", "page": 1},
            ],
            fraud_probability=0.9,
        )
        # PMLA, RBI, FIU-IND are legislative refs — no G4
        g4_warnings = [w for w in result.warnings if "G4" in w]
        assert len(g4_warnings) == 0

    def test_guardrails_summary_format(self):
        """GuardrailsResult.summary() returns a string."""
        from agent.guardrails import validate_str_draft

        result = validate_str_draft(
            str_text="text", tool_call_count=5,
            citations=[{"text": "t", "source": "s", "page": 1}],
            fraud_probability=0.9,
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "PASSED" in summary or "FAILED" in summary


# ------------------------------------------------------------------
# Risk tier mapping
# ------------------------------------------------------------------

class TestRiskTierMapping:
    """Verify the four-tier risk classification boundaries."""

    def test_low_tier(self):
        from scoring.risk_engine import get_risk_tier
        assert get_risk_tier(0.0) == "LOW"
        assert get_risk_tier(0.15) == "LOW"
        assert get_risk_tier(0.299) == "LOW"

    def test_medium_tier(self):
        from scoring.risk_engine import get_risk_tier
        assert get_risk_tier(0.3) == "MEDIUM"
        assert get_risk_tier(0.45) == "MEDIUM"

    def test_high_tier(self):
        from scoring.risk_engine import get_risk_tier
        assert get_risk_tier(0.6) == "HIGH"
        assert get_risk_tier(0.75) == "HIGH"

    def test_critical_tier(self):
        from scoring.risk_engine import get_risk_tier
        assert get_risk_tier(0.8) == "CRITICAL"
        assert get_risk_tier(1.0) == "CRITICAL"

