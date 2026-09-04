"""Independent, ground-truth-based reconciliation evaluation."""

from __future__ import annotations

from typing import Any

from .models import Decision, FinancialRecord


def evaluate(decisions: list[Decision], records: list[FinancialRecord], ground_truth: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    """Calculate transparent match, error, exception, and processing metrics."""
    invoice_ids = set(ground_truth["relationships"])
    relevant = [item for item in decisions if item.source_record_id in invoice_ids]
    correct = sum(item.decision == "MATCH" and item.matched_record_id in ground_truth["relationships"][item.source_record_id] for item in relevant)
    incorrect = sum(item.decision == "MATCH" and item.matched_record_id not in ground_truth["relationships"][item.source_record_id] for item in relevant)
    matched, total = correct + incorrect, len(relevant)
    exceptions = total - matched
    amounts = {item.id: item.amount for item in records}
    unresolved = sum(amounts[item.source_record_id] for item in relevant if item.decision == "EXCEPTION")
    return {"total_records": total, "correct_matches": correct, "incorrect_matches": incorrect, "match_rate": round(matched / total, 4) if total else 0.0, "precision": round(correct / matched, 4) if matched else 0.0, "recall": round(correct / total, 4) if total else 0.0, "false_match_rate": round(incorrect / total, 4) if total else 0.0, "exception_rate": round(exceptions / total, 4) if total else 0.0, "unresolved_financial_value": round(unresolved, 2), "throughput_records_per_second": round(total / elapsed_seconds, 2) if elapsed_seconds else 0.0, "total_processing_time_seconds": round(elapsed_seconds, 4), "automatic_matches": sum(item.method == "deterministic" and item.decision == "MATCH" for item in relevant), "ai_assisted_matches": sum(item.method == "ai_review" and item.decision == "MATCH" for item in relevant), "human_review_exceptions": sum(item.requires_human_review for item in relevant)}
