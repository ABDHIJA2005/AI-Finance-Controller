"""Deterministic normalization, candidate generation, and scoring."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

from .models import Candidate, FinancialRecord

DEFAULT_WEIGHTS: dict[str, float] = {"amount": 0.40, "date": 0.20, "reference": 0.20, "description": 0.10, "currency": 0.10}


def normalize_text(value: str) -> str:
    """Normalize text by case-folding, punctuation removal, and whitespace collapse."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.casefold())).strip()


class DeterministicMatcher:
    """Scores invoice-to-payment candidates without an LLM or future knowledge."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Create matcher and validate the configurable component weights."""
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        if set(self.weights) != set(DEFAULT_WEIGHTS) or abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("weights must contain all five components and sum to 1.0")

    def candidates(self, record: FinancialRecord, universe: Iterable[FinancialRecord], limit: int = 5) -> list[Candidate]:
        """Generate and rank a limited candidate set from other source systems."""
        scored: list[Candidate] = []
        for other in universe:
            if other.id == record.id or other.source == record.source:
                continue
            components = self._components(record, other)
            score = sum(self.weights[name] * value for name, value in components.items())
            if score >= 0.35:
                scored.append(Candidate(other.id, round(score, 4), components))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    @staticmethod
    def _components(left: FinancialRecord, right: FinancialRecord) -> dict[str, float]:
        amount_delta = abs(left.amount - right.amount) / max(left.amount, right.amount, 1.0)
        amount = max(0.0, 1.0 - amount_delta)
        day_gap = abs((left.transaction_date - right.transaction_date).days)
        date_score = max(0.0, 1.0 - day_gap / 7)
        reference = SequenceMatcher(None, normalize_text(left.reference), normalize_text(right.reference)).ratio()
        description = SequenceMatcher(None, normalize_text(left.description), normalize_text(right.description)).ratio()
        return {"amount": round(amount, 4), "date": round(date_score, 4), "reference": round(reference, 4), "description": round(description, 4), "currency": float(left.currency == right.currency)}
