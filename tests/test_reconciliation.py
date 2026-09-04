"""Tests for the seeded, explainable reconciliation workflow."""

from src.reconciliation.generator import generate_synthetic_data
from src.reconciliation.matcher import DeterministicMatcher
from src.reconciliation.service import ReconciliationService


def test_seeded_data_is_reproducible_and_multi_source() -> None:
    """The demo creates enough deterministically repeatable source records."""
    left, truth_left = generate_synthetic_data()
    right, truth_right = generate_synthetic_data()
    assert len(left) >= 100
    assert [item.to_dict() for item in left] == [item.to_dict() for item in right]
    assert truth_left == truth_right
    assert {item.source for item in left} == {"bank", "gateway", "invoice", "ledger"}


def test_exact_candidate_scores_highest() -> None:
    """An exact invoice relationship receives a high deterministic score."""
    records, _ = generate_synthetic_data()
    invoice = next(item for item in records if item.id == "INV-0001")
    candidates = DeterministicMatcher().candidates(invoice, records)
    assert candidates[0].score >= 0.95


def test_end_to_end_creates_audits_exceptions_and_metrics(tmp_path) -> None:
    """A full run produces truthful counts and traceable exception outcomes."""
    service = ReconciliationService(tmp_path)
    service.generate_data()
    status = service.reconcile()
    assert status["processed"] == 120
    assert status["exceptions"] > 0
    assert status["metrics"]["total_records"] == 120
    assert "false_match_rate" in status["metrics"]
    assert service.audits["INV-0001"].final_decision["source_record_id"] == "INV-0001"
