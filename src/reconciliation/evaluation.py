"""Independent, ground-truth-based reconciliation evaluation.

Mathematical Definitions
------------------------
1. Precision:
   Numerator:   Number of system MATCH decisions that correctly match a true counterpart ID.
   Denominator: Total number of system MATCH decisions (correct_matches + incorrect_matches).
   Formula:     correct_matches / max(total_system_matches, 1)

2. Recall (Reconcilable Universe):
   Numerator:   Number of system MATCH decisions that correctly match a true counterpart ID.
   Denominator: Total number of invoices in ground_truth that actually HAVE a true counterpart
                record in the generated universe (i.e. len(relationships[inv]) > 1).
   Formula:     correct_matches / max(reconcilable_ground_truth_invoices, 1)

3. False-Match Rate:
   Numerator:   Number of system MATCH decisions where the matched ID was NOT a true counterpart.
   Denominator: Total number of invoices processed.
   Formula:     incorrect_matches / total_records

4. Exception Rate:
   Numerator:   Number of invoices routed to human review exceptions.
   Denominator: Total number of invoices processed.
   Formula:     exceptions / total_records
"""

from __future__ import annotations

from typing import Any, Sequence

from .models import Decision, FinancialRecord


def evaluate(
    decisions: Sequence[Decision],
    records: Sequence[FinancialRecord],
    ground_truth: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Calculate transparent match, error, exception, recall, and throughput metrics."""
    invoice_ids = set(ground_truth.get("relationships", {}))
    relevant = [d for d in decisions if d.source_record_id in invoice_ids]

    correct_matches = 0
    incorrect_matches = 0

    for d in relevant:
        if d.decision == "MATCH":
            true_relations = set(ground_truth["relationships"].get(d.source_record_id, []))
            # The matched ID must be in the true relationships and cannot be the source record itself
            if d.matched_record_id in true_relations and d.matched_record_id != d.source_record_id:
                correct_matches += 1
            else:
                incorrect_matches += 1

    total_matches = correct_matches + incorrect_matches
    total_records = len(relevant)
    exceptions = total_records - total_matches

    # Ground-truth reconcilable records (invoices that have at least one valid counterpart outside themselves)
    reconcilable_invoices = sum(
        1 for inv_id in invoice_ids
        if len([r for r in ground_truth["relationships"].get(inv_id, []) if r != inv_id]) > 0
    )

    precision = round(correct_matches / total_matches, 4) if total_matches > 0 else 0.0
    recall = round(correct_matches / reconcilable_invoices, 4) if reconcilable_invoices > 0 else 0.0
    match_rate = round(total_matches / total_records, 4) if total_records > 0 else 0.0
    false_match_rate = round(incorrect_matches / total_records, 4) if total_records > 0 else 0.0
    exception_rate = round(exceptions / total_records, 4) if total_records > 0 else 0.0

    amounts = {r.id: r.amount for r in records}
    unresolved_financial_value = sum(
        amounts.get(d.source_record_id, 0.0) for d in relevant if d.decision == "EXCEPTION"
    )

    # Method-specific breakdown (4 clear outcomes)
    automatic_matches = sum(1 for d in relevant if d.decision == "MATCH" and d.method == "deterministic")
    live_llm_matches = sum(1 for d in relevant if d.decision == "MATCH" and d.method == "live_llm")
    ai_assisted_matches = live_llm_matches
    human_review_exceptions = sum(1 for d in relevant if d.decision == "EXCEPTION" or d.requires_human_review)

    throughput = round(total_records / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0

    return {
        "total_records": total_records,
        "correct_matches": correct_matches,
        "incorrect_matches": incorrect_matches,
        "automatic_matches": automatic_matches,
        "live_llm_matches": live_llm_matches,
        "ai_assisted_matches": ai_assisted_matches,
        "human_review_exceptions": human_review_exceptions,
        "match_rate": match_rate,
        "precision": precision,
        "recall": recall,
        "recall_numerator": correct_matches,
        "recall_denominator": reconcilable_invoices,
        "recall_definition": "correct_matches / reconcilable_invoices_in_ground_truth",
        "false_match_rate": false_match_rate,
        "exception_rate": exception_rate,
        "unresolved_financial_value": round(unresolved_financial_value, 2),
        "throughput_records_per_second": throughput,
        "total_processing_time_seconds": round(elapsed_seconds, 4),
    }
