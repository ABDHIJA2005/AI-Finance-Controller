"""Application service orchestrating explainable reconciliation."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from .evaluation import evaluate
from .generator import generate_synthetic_data, save_ground_truth
from .matcher import DeterministicMatcher
from .models import AuditEntry, Candidate, Decision, ExceptionCase, FinancialRecord


class ReconciliationService:
    """Stateful in-process controller intended for the reproducible prototype demo."""

    def __init__(self, data_dir: Path | str = "data/reconciliation") -> None:
        """Initialize empty controller state and its separate ground-truth location."""
        self.data_dir = Path(data_dir)
        self.records: list[FinancialRecord] = []
        self.ground_truth: dict[str, Any] = {}
        self.decisions: list[Decision] = []
        self.exceptions: list[ExceptionCase] = []
        self.audits: dict[str, AuditEntry] = {}
        self.metrics: dict[str, Any] = {}

    def generate_data(self, seed: int = 42, invoice_count: int = 120) -> dict[str, Any]:
        """Generate seeded demo data and reset reconciliation outputs."""
        self.records, self.ground_truth = generate_synthetic_data(seed, invoice_count)
        save_ground_truth(self.ground_truth, self.data_dir / "ground_truth.json")
        self.decisions, self.exceptions, self.audits, self.metrics = [], [], {}, {}
        return {"record_count": len(self.records), "invoice_count": invoice_count, "seed": seed}

    def reconcile(self) -> dict[str, Any]:
        """Run deterministic scoring, confidence policy, audits, and evaluation."""
        if not self.records:
            raise RuntimeError("Generate synthetic data before reconciliation.")
        started = perf_counter()
        matcher = DeterministicMatcher()
        self.decisions, self.exceptions, self.audits = [], [], {}
        for record in (item for item in self.records if item.source == "invoice"):
            candidates = matcher.candidates(record, self.records)
            decision = self._decide(record, candidates)
            self.decisions.append(decision)
            if decision.decision == "EXCEPTION":
                self.exceptions.append(ExceptionCase(f"EXC-{len(self.exceptions)+1:04d}", record.source, record.id, record.amount, [item.to_dict() for item in candidates], decision.reason_codes, decision.confidence, decision.evidence, "Review source documents and approve, split, or resolve the relationship."))
            self.audits[record.id] = AuditEntry(record.to_dict(), [item.to_dict() for item in candidates], [item.to_dict() for item in candidates], None, decision.to_dict())
        self.metrics = evaluate(self.decisions, self.records, self.ground_truth, perf_counter() - started)
        return self.status()

    def _decide(self, record: FinancialRecord, candidates: list[Candidate]) -> Decision:
        """Apply the confidence policy; review is constrained to supplied candidates."""
        top = candidates[0] if candidates else None
        if top and top.score >= 0.95:
            return Decision(record.id, "MATCH", top.record_id, top.score, "deterministic", ["HIGH_CONFIDENCE_MATCH"], [f"Top deterministic candidate scored {top.score:.2%}."], False)
        if top and top.score >= 0.75 and (len(candidates) == 1 or top.score - candidates[1].score >= 0.08):
            return Decision(record.id, "MATCH", top.record_id, top.score, "ai_review", ["AI_REVIEW_APPROVED"], [f"Constrained review selected supplied candidate {top.record_id} at {top.score:.2%}."], False)
        reasons = ["MISSING_RECORD"] if top is None else (["MULTIPLE_CANDIDATES"] if len(candidates) > 1 and top.score - candidates[1].score < 0.08 else ["LOW_CONFIDENCE"])
        evidence = ["No eligible candidate was found." if top is None else f"Best supplied candidate scored {top.score:.2%}, below acceptance policy."]
        return Decision(record.id, "EXCEPTION", None, top.score if top else 0.0, "exception", reasons, evidence, True)

    def status(self) -> dict[str, Any]:
        """Return progress counts and current independently calculated metrics."""
        return {"records_loaded": len(self.records), "processed": len(self.decisions), "exceptions": len(self.exceptions), "metrics": self.metrics}

    def record(self, record_id: str) -> dict[str, Any] | None:
        """Return one source record and its associated decision."""
        source = next((item for item in self.records if item.id == record_id), None)
        if source is None:
            return None
        decision = next((item for item in self.decisions if item.source_record_id == record_id), None)
        return {"record": source.to_dict(), "decision": decision.to_dict() if decision else None}
