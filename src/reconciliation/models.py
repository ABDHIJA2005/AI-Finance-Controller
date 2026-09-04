"""Typed domain models and strict Pydantic validation schemas for reconciliation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

STANDARD_REASON_CODES = {
    "MISSING_RECORD",
    "DUPLICATE",
    "AMOUNT_MISMATCH",
    "DATE_MISMATCH",
    "REFERENCE_MISMATCH",
    "MULTIPLE_CANDIDATES",
    "LOW_CONFIDENCE",
    "UNKNOWN",
    "HIGH_CONFIDENCE_MATCH",
    "AI_REVIEW_APPROVED",
    "AI_REVIEW_UNAVAILABLE",
}


@dataclass(frozen=True)
class FinancialRecord:
    """A normalized financial record from one operational source."""

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


class AIReviewDecision(BaseModel):
    """Strict Pydantic schema for AI-assisted reconciliation decisions."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["MATCH", "EXCEPTION"]
    matched_record_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    requires_human_review: bool

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> AIReviewDecision:
        """Validate relational consistency between decision and matched target."""
        if self.decision == "MATCH":
            if not self.matched_record_id or not self.matched_record_id.strip():
                raise ValueError("MATCH decision requires a non-empty matched_record_id.")
        elif self.decision == "EXCEPTION":
            if self.matched_record_id is not None:
                raise ValueError("EXCEPTION decision must have matched_record_id set to None.")

        # Ensure reason codes are valid known tokens
        for code in self.reason_codes:
            if code not in STANDARD_REASON_CODES:
                raise ValueError(f"Unknown reason_code: '{code}'. Must be one of {sorted(STANDARD_REASON_CODES)}")

        return self


@dataclass
class Decision:
    """Validated reconciliation outcome for a source record."""

    source_record_id: str
    decision: Literal["MATCH", "EXCEPTION"]
    matched_record_id: str | None
    confidence: float
    method: Literal["deterministic", "live_llm", "exception"]
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
    ai_interaction: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit entry."""
        return asdict(self)
