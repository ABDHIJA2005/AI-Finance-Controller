"""Comprehensive unit and integration tests for the AI Finance Controller."""

from __future__ import annotations

from datetime import date
import pytest
from pydantic import ValidationError

from src.reconciliation.ai_reviewer import AIReviewer
from src.reconciliation.evaluation import evaluate
from src.reconciliation.generator import generate_synthetic_data
from src.reconciliation.matcher import DeterministicMatcher
from src.reconciliation.models import AIReviewDecision, Candidate, Decision, FinancialRecord
from src.reconciliation.service import ReconciliationService


def test_seeded_data_is_reproducible_and_multi_source() -> None:
    """The demo creates deterministically repeatable source records across all 4 systems."""
    left, truth_left = generate_synthetic_data(seed=42, invoice_count=120)
    right, truth_right = generate_synthetic_data(seed=42, invoice_count=120)
    assert len(left) >= 100
    assert [item.to_dict() for item in left] == [item.to_dict() for item in right]
    assert truth_left == truth_right
    assert {item.source for item in left} == {"bank", "gateway", "invoice", "ledger"}


def test_exact_candidate_scores_highest() -> None:
    """An exact invoice relationship receives a high deterministic score (>= 0.95)."""
    records, _ = generate_synthetic_data()
    invoice = next(item for item in records if item.id == "INV-0001")
    candidates = DeterministicMatcher().candidates(invoice, records)
    assert len(candidates) > 0
    assert candidates[0].score >= 0.95


def test_routing_exact_match_to_deterministic_match() -> None:
    """Score >= 0.95 with single clear candidate routes to automatic deterministic match."""
    service = ReconciliationService()
    inv = FinancialRecord("INV-0101", "invoice", 5000.0, "INR", date(2026, 1, 15), "TXN-0101", "Invoice from Vendor")
    bnk = FinancialRecord("BNK-0101", "bank", 5000.0, "INR", date(2026, 1, 15), "TXN-0101", "Credit Vendor")
    service.records = [inv, bnk]
    service.ground_truth = {"relationships": {"INV-0101": ["INV-0101", "BNK-0101"]}}

    service.reconcile()
    dec = service.decisions[0]
    assert dec.decision == "MATCH"
    assert dec.method == "deterministic"
    assert dec.confidence >= 0.95
    assert dec.matched_record_id == "BNK-0101"
    assert "HIGH_CONFIDENCE_MATCH" in dec.reason_codes


def test_unconfigured_ai_routes_ambiguous_record_to_exception() -> None:
    """Missing live credentials never create a simulated AI match."""
    reviewer = AIReviewer(api_key=None)
    inv = FinancialRecord("INV-0102", "invoice", 10000.0, "INR", date(2026, 1, 15), "TXN-0102", "Invoice from Vendor")
    # 1.8% gateway fee variance: score lands in [0.75, 0.95)
    gw = FinancialRecord("GW-0102", "gateway", 9820.0, "INR", date(2026, 1, 15), "TXN-0102", "Settlement Vendor")
    records = [inv, gw]

    matcher = DeterministicMatcher()
    candidates = matcher.candidates(inv, records)
    assert 0.75 <= candidates[0].score < 0.95

    decision, audit = reviewer.review_ambiguous_record(inv, candidates, records)
    assert decision.source_record_id == "INV-0102"
    assert audit["mode"] == "unconfigured"
    assert decision.decision == "EXCEPTION"
    assert "AI_REVIEW_UNAVAILABLE" in decision.reason_codes
    assert len(audit["tools_called"]) > 0
    assert any(t["tool"] == "compare_records" for t in audit["tools_called"])


def test_routing_low_confidence_to_exception() -> None:
    """Records with score < 0.75 route directly to exception without forcing match."""
    service = ReconciliationService()
    inv = FinancialRecord("INV-0103", "invoice", 10000.0, "INR", date(2026, 1, 15), "TXN-0103", "Invoice A")
    # Completely different amount, date, reference, description
    bnk = FinancialRecord("BNK-0999", "bank", 200.0, "INR", date(2026, 3, 20), "REF-9999", "Other B")
    service.records = [inv, bnk]
    service.ground_truth = {"relationships": {"INV-0103": ["INV-0103"]}}

    service.reconcile()
    dec = service.decisions[0]
    assert dec.decision == "EXCEPTION"
    assert dec.matched_record_id is None
    assert dec.requires_human_review is True
    assert dec.confidence < 0.75


def test_multiple_candidates_duplicate_ambiguity_not_auto_matched() -> None:
    """Score >= 0.95 with duplicate identical candidates does NOT auto-match."""
    service = ReconciliationService()
    inv = FinancialRecord("INV-0104", "invoice", 5000.0, "INR", date(2026, 1, 15), "TXN-0104", "Invoice Vendor")
    b1 = FinancialRecord("BNK-0104", "bank", 5000.0, "INR", date(2026, 1, 15), "TXN-0104", "Credit Vendor")
    b2 = FinancialRecord("BNK-DUP-0104", "bank", 5000.0, "INR", date(2026, 1, 15), "TXN-0104", "Credit Vendor")
    service.records = [inv, b1, b2]
    service.ground_truth = {"relationships": {"INV-0104": ["INV-0104", "BNK-0104", "BNK-DUP-0104"]}}

    service.reconcile()
    dec = service.decisions[0]
    assert dec.decision == "EXCEPTION"
    assert "MULTIPLE_CANDIDATES" in dec.reason_codes
    assert dec.requires_human_review is True


def test_ai_cannot_invent_candidate() -> None:
    """An AI review decision selecting a record not in supplied candidates is rejected."""
    reviewer = AIReviewer()
    inv = FinancialRecord("INV-0105", "invoice", 5000.0, "INR", date(2026, 1, 15), "TXN-0105", "Invoice Vendor")
    cand = Candidate("GW-0105", 0.85, {"amount": 0.9, "date": 1.0, "reference": 1.0, "description": 0.5, "currency": 1.0})
    audit_log = {}

    # Raw response attempts to match an invented candidate ID
    invented_data = {
        "decision": "MATCH",
        "matched_record_id": "BNK-GHOST-9999",
        "confidence": 0.92,
        "reason_codes": ["AI_REVIEW_APPROVED"],
        "evidence": ["Matched invented transaction"],
        "requires_human_review": False,
    }
    decision = reviewer._validate_and_build_decision(
        record=inv,
        candidates=[cand],
        parsed_data=invented_data,
        audit_log=audit_log,
    )
    assert decision.decision == "EXCEPTION"
    assert decision.matched_record_id is None
    assert audit_log["candidate_containment_validation"]["status"] == "FAILED"


def test_invalid_ai_json_schema_rejected() -> None:
    """Invalid AI JSON output is rejected by Pydantic and creates an exception."""
    reviewer = AIReviewer()
    inv = FinancialRecord("INV-0106", "invoice", 5000.0, "INR", date(2026, 1, 15), "TXN-0106", "Invoice Vendor")
    cand = Candidate("GW-0106", 0.85, {})
    audit_log = {}

    # Invalid: MATCH decision with matched_record_id = None, and confidence out of bounds
    invalid_data = {
        "decision": "MATCH",
        "matched_record_id": None,
        "confidence": 1.5,
        "reason_codes": ["INVALID_CODE"],
        "evidence": [],
        "requires_human_review": False,
    }
    decision = reviewer._validate_and_build_decision(
        record=inv,
        candidates=[cand],
        parsed_data=invalid_data,
        audit_log=audit_log,
    )
    assert decision.decision == "EXCEPTION"
    assert audit_log["pydantic_validation"]["status"] == "FAILED"


def test_ai_match_rejected_when_evidence_cites_unsupplied_records() -> None:
    """An AI MATCH decision is rejected if its evidence references unsupplied records."""
    reviewer = AIReviewer()
    inv = FinancialRecord("INV-0107", "invoice", 5000.0, "INR", date(2026, 1, 15), "TXN-0107", "Invoice Vendor")
    cand = Candidate("GW-0107", 0.88, {})
    audit_log = {}

    hallucinated_evidence_data = {
        "decision": "MATCH",
        "matched_record_id": "GW-0107",
        "confidence": 0.88,
        "reason_codes": ["AI_REVIEW_APPROVED"],
        "evidence": ["Matched because BNK-9999 confirms bank settlement"],  # BNK-9999 is unsupplied!
        "requires_human_review": False,
    }
    decision = reviewer._validate_and_build_decision(
        record=inv,
        candidates=[cand],
        parsed_data=hallucinated_evidence_data,
        audit_log=audit_log,
    )
    assert decision.decision == "EXCEPTION"
    assert decision.matched_record_id is None
    assert audit_log["evidence_validation"]["status"] == "FAILED"


def test_pydantic_ai_decision_validation_bounds() -> None:
    """Test explicit Pydantic schema validation for AIReviewDecision."""
    # Valid MATCH
    valid = AIReviewDecision(
        decision="MATCH",
        matched_record_id="GW-0001",
        confidence=0.88,
        reason_codes=["AI_REVIEW_APPROVED"],
        evidence=["Valid candidate match"],
        requires_human_review=False,
    )
    assert valid.confidence == 0.88

    # Confidence out of bounds
    with pytest.raises(ValidationError):
        AIReviewDecision(
            decision="MATCH",
            matched_record_id="GW-0001",
            confidence=1.2,
            reason_codes=["AI_REVIEW_APPROVED"],
            evidence=["Variance noted"],
            requires_human_review=False,
        )

    # Unknown reason code
    with pytest.raises(ValidationError):
        AIReviewDecision(
            decision="MATCH",
            matched_record_id="GW-0001",
            confidence=0.88,
            reason_codes=["FABRICATED_CODE"],
            evidence=["Variance noted"],
            requires_human_review=False,
        )


def test_independent_evaluation_metrics_and_recall_formula() -> None:
    """Evaluation engine calculates truthful precision, recall with explicit formula, and throughput."""
    records = [
        FinancialRecord("INV-1", "invoice", 1000.0, "INR", date(2026, 1, 1), "T-1", "Inv 1"),
        FinancialRecord("INV-2", "invoice", 2000.0, "INR", date(2026, 1, 1), "T-2", "Inv 2"),
        FinancialRecord("INV-3", "invoice", 3000.0, "INR", date(2026, 1, 1), "T-3", "Inv 3"),  # missing counterpart
    ]
    ground_truth = {
        "relationships": {
            "INV-1": ["INV-1", "BNK-1"],
            "INV-2": ["INV-2", "BNK-2"],
            "INV-3": ["INV-3"],  # unreconcilable
        }
    }
    decisions = [
        Decision("INV-1", "MATCH", "BNK-1", 0.98, "deterministic", ["HIGH_CONFIDENCE_MATCH"], ["ok"], False),
        Decision("INV-2", "MATCH", "BNK-999", 0.80, "live_llm", ["AI_REVIEW_APPROVED"], ["bad"], False), # false match
        Decision("INV-3", "EXCEPTION", None, 0.0, "exception", ["MISSING_RECORD"], ["missing"], True),
    ]

    metrics = evaluate(decisions, records, ground_truth, elapsed_seconds=0.05)
    assert metrics["total_records"] == 3
    assert metrics["correct_matches"] == 1
    assert metrics["incorrect_matches"] == 1
    assert metrics["precision"] == 0.5  # 1 / 2 matches
    assert metrics["recall"] == 0.5     # 1 / 2 reconcilable ground truth
    assert metrics["recall_numerator"] == 1
    assert metrics["recall_denominator"] == 2
    assert metrics["false_match_rate"] == round(1 / 3, 4)
    assert metrics["unresolved_financial_value"] == 3000.0
    assert metrics["automatic_matches"] == 1
    assert metrics["live_llm_matches"] == 1
    assert metrics["human_review_exceptions"] == 1


def test_seeded_end_to_end_demo_run(tmp_path) -> None:
    """Full seeded demo run produces complete pipeline stages, honest exceptions, and audit entries."""
    service = ReconciliationService(tmp_path)
    service.generate_data(seed=42, invoice_count=120)
    status = service.reconcile()

    assert status["records_loaded"] >= 200
    assert status["processed"] == 120
    assert status["exceptions"] > 0
    assert status["metrics"]["total_records"] == 120
    assert status["metrics"]["precision"] >= 0.95
    assert status["metrics"]["recall"] > 0.60
    assert status["metrics"]["unresolved_financial_value"] > 0
    assert "pipeline_stages" in status
    assert len(status["pipeline_stages"]) >= 6

    # Verify audit trail contains complete interaction
    inv_id = "INV-0001"
    audit = service.audits[inv_id]
    assert audit.source_record["id"] == inv_id
    assert len(audit.candidates) > 0
    assert audit.final_decision["source_record_id"] == inv_id
