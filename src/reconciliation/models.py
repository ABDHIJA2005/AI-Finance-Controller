"""Typed domain models for the reconciliation controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any


@dataclass(frozen=True)
class FinancialRecord:
    """A normalized financial record from one operational source.

    Parameters
    ----------
    id : str
        Stable source-qualified record identifier.
    source : str
        Origin system name.
    amount : float
        Positive transaction amount in the stated currency.
    currency : str
        ISO 4217 currency code.
    transaction_date : date
        Business date recorded by the source.
    reference : str
        Source reference, potentially corrupted in synthetic data.
    description : str
        Human-readable transaction text.
    """

    id: str
    source: str
    amount: float
    currency: str
    transaction_date: date
    reference: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record into JSON-compatible primitive values."""
        result = asdict(self)
        result["transaction_date"] = self.transaction_date.isoformat()
        return result


@dataclass(frozen=True)
class Candidate:
    """A scored candidate relationship for one source record."""

    record_id: str
    score: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize candidate scoring details."""
        return asdict(self)


@dataclass
class Decision:
    """Validated reconciliation outcome for a source record."""

    source_record_id: str
    decision: str
    matched_record_id: str | None
    confidence: float
    method: str
    reason_codes: list[str]
    evidence: list[str]
    requires_human_review: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize this outcome."""
        return asdict(self)


@dataclass
class ExceptionCase:
    """An unresolved reconciliation item retained for operational review."""

    id: str
    source: str
    source_record_id: str
    amount: float
    candidate_matches: list[dict[str, Any]]
    reason_codes: list[str]
    confidence: float
    evidence: list[str]
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize this exception."""
        return asdict(self)


@dataclass
class AuditEntry:
    """Immutable explanation of inputs and outcome for one decision."""

    source_record: dict[str, Any]
    candidates: list[dict[str, Any]]
    deterministic_scores: list[dict[str, Any]]
    ai_decision: dict[str, Any] | None
    final_decision: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit entry."""
        return asdict(self)
