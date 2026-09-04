"""Application service orchestrating explainable reconciliation."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from .ai_reviewer import AIReviewer
from .evaluation import evaluate
from .generator import generate_synthetic_data, save_ground_truth
from .matcher import DeterministicMatcher
from .models import AuditEntry, Candidate, Decision, ExceptionCase, FinancialRecord


class ReconciliationService:
    """Stateful in-process controller orchestrating the multi-stage reconciliation pipeline."""

    def __init__(self, data_dir: Path | str = "data/reconciliation", ai_mode: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.ai_reviewer = AIReviewer()
        self.records: list[FinancialRecord] = []
        self.ground_truth: dict[str, Any] = {}
        self.decisions: list[Decision] = []
        self.exceptions: list[ExceptionCase] = []
        self.audits: dict[str, AuditEntry] = {}
        self.metrics: dict[str, Any] = {}
        self.activity_log: list[str] = []
        self.pipeline_stages: list[dict[str, Any]] = []

    def generate_data(self, seed: int = 42, invoice_count: int = 120) -> dict[str, Any]:
        """Generate seeded demo data and reset reconciliation outputs."""
        self.records, self.ground_truth = generate_synthetic_data(seed, invoice_count)
        save_ground_truth(self.ground_truth, self.data_dir / "ground_truth.json")
        self.decisions, self.exceptions, self.audits, self.metrics = [], [], {}, {}
        self.activity_log = [
            f"Ingested {len(self.records)} financial records across 4 operational systems (Invoices: {invoice_count}, Seed: {seed})."
        ]
        self.pipeline_stages = []
        return {"record_count": len(self.records), "invoice_count": invoice_count, "seed": seed}

    def reconcile(self) -> dict[str, Any]:
        """Run the full specification-compliant multi-stage reconciliation pipeline.

        Pipeline Stages
        ---------------
        DATA INGESTION
        -> NORMALIZATION
        -> EXACT MATCHING & CANDIDATE GENERATION
        -> RULE/FUZZY SCORING & ROUTING
        -> AI REVIEW FOR AMBIGUOUS CASES
        -> VALIDATION (Pydantic schema + candidate containment)
        -> MATCH / EXCEPTION ROUTING
        -> AUDIT LOG
        -> EVALUATION
        """
        if not self.records:
            raise RuntimeError("Generate synthetic data before reconciliation.")

        started = perf_counter()
        matcher = DeterministicMatcher()
        self.decisions, self.exceptions, self.audits = [], [], {}
        self.activity_log = []

        invoices = [r for r in self.records if r.source == "invoice"]
        self.activity_log.append(f"Stage 1 [Ingestion]: Loaded {len(self.records)} records ({len(invoices)} invoices to reconcile).")
        self.activity_log.append("Stage 2 [Normalization]: Canonicalized amounts, ISO-8601 dates, currency, and reference strings.")

        deterministic_count = 0
        ai_review_count = 0
        direct_exception_count = 0

        for record in invoices:
            candidates = matcher.candidates(record, self.records)
            top = candidates[0] if candidates else None

            # Rule 1: No candidates or very low score (< 0.75) -> direct Exception
            if top is None or top.score < 0.75:
                direct_exception_count += 1
                reason = "MISSING_RECORD" if top is None else "LOW_CONFIDENCE"
                evidence = ["No candidate records found above 35% similarity threshold."] if top is None else [
                    f"Top candidate {top.record_id} scored {top.score:.2%}, below 75% threshold."
                ]
                decision = Decision(
                    source_record_id=record.id,
                    decision="EXCEPTION",
                    matched_record_id=None,
                    confidence=top.score if top else 0.0,
                    method="exception",
                    reason_codes=[reason],
                    evidence=evidence,
                    requires_human_review=True,
                )
                self.decisions.append(decision)
                self.exceptions.append(self._build_exception_case(record, candidates, decision, "Review statement feeds for missing or unrecorded transaction."))
                self.audits[record.id] = AuditEntry(
                    source_record=record.to_dict(),
                    candidates=[c.to_dict() for c in candidates],
                    deterministic_scores=[c.to_dict() for c in candidates],
                    ai_decision=None,
                    final_decision=decision.to_dict(),
                    ai_interaction=None,
                )

            # Rule 2: Multiple candidates ambiguity (even if top >= 0.95, cannot auto-match duplicate ambiguity!)
            elif len(candidates) > 1 and abs(top.score - candidates[1].score) < 0.05:
                ai_review_count += 1
                # Route through AI Reviewer for disambiguation / validation
                decision, ai_interaction = self.ai_reviewer.review_ambiguous_record(record, candidates, self.records)
                self.decisions.append(decision)
                if decision.decision == "EXCEPTION":
                    self.exceptions.append(self._build_exception_case(record, candidates, decision, "Manual operator review required to disambiguate between multiple matching entries."))
                self.audits[record.id] = AuditEntry(
                    source_record=record.to_dict(),
                    candidates=[c.to_dict() for c in candidates],
                    deterministic_scores=[c.to_dict() for c in candidates],
                    ai_decision=ai_interaction.get("raw_ai_response"),
                    final_decision=decision.to_dict(),
                    ai_interaction=ai_interaction,
                )

            # Rule 3: Single clear top candidate with score >= 0.95 -> deterministic automatic match
            elif top.score >= 0.95:
                deterministic_count += 1
                decision = Decision(
                    source_record_id=record.id,
                    decision="MATCH",
                    matched_record_id=top.record_id,
                    confidence=top.score,
                    method="deterministic",
                    reason_codes=["HIGH_CONFIDENCE_MATCH"],
                    evidence=[f"Top deterministic candidate {top.record_id} scored {top.score:.2%} with distinct separation."],
                    requires_human_review=False,
                )
                self.decisions.append(decision)
                self.audits[record.id] = AuditEntry(
                    source_record=record.to_dict(),
                    candidates=[c.to_dict() for c in candidates],
                    deterministic_scores=[c.to_dict() for c in candidates],
                    ai_decision=None,
                    final_decision=decision.to_dict(),
                    ai_interaction=None,
                )

            # Rule 4: Ambiguous score range (0.75 <= score < 0.95) -> route strictly to AI Review
            else:
                ai_review_count += 1
                decision, ai_interaction = self.ai_reviewer.review_ambiguous_record(record, candidates, self.records)
                self.decisions.append(decision)
                if decision.decision == "EXCEPTION":
                    self.exceptions.append(self._build_exception_case(record, candidates, decision, "Verify supporting documents for amount/date variance."))
                self.audits[record.id] = AuditEntry(
                    source_record=record.to_dict(),
                    candidates=[c.to_dict() for c in candidates],
                    deterministic_scores=[c.to_dict() for c in candidates],
                    ai_decision=ai_interaction.get("raw_ai_response"),
                    final_decision=decision.to_dict(),
                    ai_interaction=ai_interaction,
                )

        elapsed = perf_counter() - started
        self.metrics = evaluate(self.decisions, self.records, self.ground_truth, elapsed)

        self.activity_log.extend([
            f"Stage 3 [Scoring]: Evaluated {len(invoices)} records across 5 weighted dimensions.",
            f"Stage 4 [Deterministic]: Automatically matched {deterministic_count} high-confidence records.",
            f"Stage 5 [AI Review]: Routed {ai_review_count} ambiguous/duplicate records to AI review layer (Mode: {self.ai_reviewer.mode}).",
            f"Stage 6 [Exceptions]: Isolated {len(self.exceptions)} total human-review exception cases.",
            f"Stage 7 [Evaluation]: Objective ground-truth evaluation finished in {elapsed:.4f}s.",
        ])

        self.pipeline_stages = [
            {"stage": "Data Ingestion", "count": len(self.records), "description": "Loaded multi-source financial records"},
            {"stage": "Normalization", "count": len(self.records), "description": "Dates, amounts, casing, reference cleaning"},
            {"stage": "Candidate Matching", "count": len(invoices), "description": "Fuzzy multi-factor candidate ranking"},
            {"stage": "Deterministic Matches", "count": self.metrics.get("automatic_matches", 0), "description": "High-confidence >=95% single matches"},
            {"stage": "AI Review Cases", "count": ai_review_count, "description": "Ambiguous 75-95% & duplicate cases reviewed by agent"},
            {"stage": "AI-Assisted Matches", "count": self.metrics.get("ai_assisted_matches", 0), "description": "Validated matches approved by AI review"},
            {"stage": "Exceptions", "count": len(self.exceptions), "description": "Unresolved records held for human controller"},
            {"stage": "Evaluation", "count": len(invoices), "description": "Independent ground-truth precision/recall audit"},
        ]

        return self.status()

    @staticmethod
    def _build_exception_case(
        record: FinancialRecord,
        candidates: list[Candidate],
        decision: Decision,
        recommended_action: str,
    ) -> ExceptionCase:
        return ExceptionCase(
            id=f"EXC-{record.id.replace('INV-', '')}",
            source=record.source,
            source_record_id=record.id,
            amount=record.amount,
            candidate_matches=[c.to_dict() for c in candidates],
            reason_codes=decision.reason_codes,
            confidence=decision.confidence,
            evidence=decision.evidence,
            recommended_action=recommended_action,
        )

    def status(self) -> dict[str, Any]:
        """Return progress counts and current independently calculated metrics."""
        return {
            "records_loaded": len(self.records),
            "processed": len(self.decisions),
            "exceptions": len(self.exceptions),
            "metrics": self.metrics,
            "pipeline_stages": self.pipeline_stages,
            "activity_log": self.activity_log,
            "ai_review": {"mode": self.ai_reviewer.mode, "model": self.ai_reviewer.model, "configured": self.ai_reviewer.mode == "live_llm"},
        }

    def record(self, record_id: str) -> dict[str, Any] | None:
        """Return one source record and its associated decision."""
        source = next((item for item in self.records if item.id == record_id), None)
        if source is None:
            return None
        decision = next((item for item in self.decisions if item.source_record_id == record_id), None)
        audit = self.audits.get(record_id)
        return {
            "record": source.to_dict(),
            "decision": decision.to_dict() if decision else None,
            "audit": audit.to_dict() if audit else None,
        }
