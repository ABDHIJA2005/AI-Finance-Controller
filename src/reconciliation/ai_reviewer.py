"""Live, guarded OpenAI review for ambiguous reconciliation candidates."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Sequence

from pydantic import ValidationError
from dotenv import load_dotenv

from .agent_tools import ReconciliationAgentTools
from .models import AIReviewDecision, Candidate, Decision, FinancialRecord, STANDARD_REASON_CODES


class AIReviewer:
    """Review ambiguous records with a live model and strict local guardrails.

    Missing credentials never degrade to simulated AI. Instead, the record is
    held as an exception so an operator can distinguish unavailable AI from a
    real model decision.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Configure the live reviewer from explicit values or environment variables."""
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("AI_MODEL", "gpt-4.1-mini")
        self.mode = "live_llm" if self.api_key else "unconfigured"

    def review_ambiguous_record(self, record: FinancialRecord, candidates: Sequence[Candidate], all_records: Sequence[FinancialRecord]) -> tuple[Decision, dict[str, Any]]:
        """Review supplied candidates only and return a locally validated decision."""
        tools = ReconciliationAgentTools(all_records, {record.id: list(candidates)})
        audit: dict[str, Any] = {"source_record": record.to_dict(), "candidates_supplied": [c.to_dict() for c in candidates], "mode": self.mode, "model": self.model, "tools_called": [], "raw_ai_response": None, "pydantic_validation": None, "candidate_containment_validation": None, "evidence_validation": None, "final_decision": None}
        if not candidates:
            decision = self._exception(record.id, 0.0, "MISSING_RECORD", "No candidate records were supplied for review.")
            tools.create_exception(record.id, decision.reason_codes, decision.confidence, decision.evidence, "Review external statement feeds for a missing transaction.")
        elif not self.api_key:
            tools.get_transaction(record.id)
            tools.get_candidates(record.id)
            comparison = tools.compare_records(record.id, candidates[0].record_id)
            tools.calculate_difference(record.amount, comparison["record_b"]["amount"])
            code = "MULTIPLE_CANDIDATES" if len(candidates) > 1 and abs(candidates[0].score - candidates[1].score) < 0.05 else "AI_REVIEW_UNAVAILABLE"
            evidence = "Multiple plausible supplied candidates require human disambiguation." if code == "MULTIPLE_CANDIDATES" else "Live AI review is not configured; this ambiguous record requires human review."
            decision = self._exception(record.id, candidates[0].score, code, evidence)
            tools.create_exception(record.id, decision.reason_codes, decision.confidence, decision.evidence, "Set OPENAI_API_KEY and rerun, or resolve this case manually.")
        else:
            try:
                raw, parsed = self._call_live_llm(record, candidates, tools)
                audit["raw_ai_response"] = raw
                decision = self._validate_and_build_decision(record, candidates, parsed, audit)
                if decision.decision == "MATCH":
                    tools.mark_match(record.id, decision.matched_record_id or "", decision.confidence, decision.evidence, decision.reason_codes)
                else:
                    tools.create_exception(record.id, decision.reason_codes, decision.confidence, decision.evidence, "Review source documents before resolving this exception.")
            except Exception as error:
                audit["raw_ai_response"] = {"error": str(error)}
                decision = self._exception(record.id, candidates[0].score, "AI_REVIEW_UNAVAILABLE", "Live AI review failed; the record was held for human review.")
                tools.create_exception(record.id, decision.reason_codes, decision.confidence, decision.evidence, "Check the AI service configuration and retry; do not force a match.")
        audit["tools_called"] = list(tools.call_log)
        audit["final_decision"] = decision.to_dict()
        return decision, audit

    def _call_live_llm(self, record: FinancialRecord, candidates: Sequence[Candidate], tools: ReconciliationAgentTools) -> tuple[dict[str, Any], dict[str, Any]]:
        """Call the Responses API with strict JSON output and bounded tool evidence."""
        from openai import OpenAI

        source = tools.get_transaction(record.id)
        supplied = tools.get_candidates(record.id)
        comparisons: list[dict[str, Any]] = []
        for candidate in candidates:
            comparison = tools.compare_records(record.id, candidate.record_id)
            comparisons.append(comparison)
            tools.calculate_difference(record.amount, comparison["record_b"]["amount"])
        tools.get_reconciliation_status()
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["decision", "matched_record_id", "confidence", "reason_codes", "evidence", "requires_human_review"],
            "properties": {
                "decision": {"type": "string", "enum": ["MATCH", "EXCEPTION"]},
                "matched_record_id": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason_codes": {"type": "array", "minItems": 1, "items": {"type": "string", "enum": sorted(STANDARD_REASON_CODES)}},
                "evidence": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "requires_human_review": {"type": "boolean"},
            },
        }
        instructions = (
            "You are a financial reconciliation reviewer. Decide only from the supplied source record, candidate IDs, and tool results. "
            "Never invent a record, reference, amount, date, or evidence. MATCH only when a supplied candidate is supported by the evidence; otherwise EXCEPTION. "
            "For MATCH, matched_record_id must be exactly one supplied candidate ID and requires_human_review must be false. "
            "For EXCEPTION, matched_record_id must be null and requires_human_review must be true. Evidence must be concise, factual, and drawn from the provided tool results."
        )
        response = OpenAI(api_key=self.api_key).responses.create(
            model=self.model, instructions=instructions,
            input=json.dumps({"source_record": source, "supplied_candidates": supplied, "comparisons": comparisons}),
            text={"format": {"type": "json_schema", "name": "reconciliation_decision", "strict": True, "schema": schema}}, store=False,
        )
        return {"response_id": response.id, "model": self.model, "output_text": response.output_text}, json.loads(response.output_text)

    @staticmethod
    def _exception(record_id: str, confidence: float, code: str, evidence: str) -> Decision:
        """Create a safe exception decision without presenting it as an AI match."""
        return Decision(record_id, "EXCEPTION", None, round(confidence, 4), "exception", [code], [evidence], True)

    def _validate_and_build_decision(self, record: FinancialRecord, candidates: Sequence[Candidate], parsed_data: dict[str, Any], audit_log: dict[str, Any]) -> Decision:
        """Apply schema, candidate-containment, and evidence-grounding validation."""
        try:
            reviewed = AIReviewDecision.model_validate(parsed_data)
            audit_log["pydantic_validation"] = {"status": "PASSED"}
        except (ValidationError, ValueError) as error:
            audit_log["pydantic_validation"] = {"status": "FAILED", "error": str(error)}
            return self._exception(record.id, 0.0, "UNKNOWN", "AI output failed strict schema validation and was rejected.")
        candidate_ids = {candidate.record_id for candidate in candidates}
        if reviewed.decision == "MATCH" and reviewed.matched_record_id not in candidate_ids:
            audit_log["candidate_containment_validation"] = {"status": "FAILED", "error": "Model selected a candidate not supplied for review."}
            return self._exception(record.id, reviewed.confidence, "UNKNOWN", "AI selected an unsupplied candidate and the decision was rejected.")
        audit_log["candidate_containment_validation"] = {"status": "PASSED"}
        evidence_ids = set(re.findall(r"\b[A-Z]{2,5}(?:-[A-Z0-9]+)*-[0-9]{4}\b", " ".join(reviewed.evidence)))
        if evidence_ids - (candidate_ids | {record.id}):
            audit_log["evidence_validation"] = {"status": "FAILED", "error": "Evidence cited an unsupplied record ID."}
            return self._exception(record.id, reviewed.confidence, "UNKNOWN", "AI cited an unsupplied record and the decision was rejected.")
        audit_log["evidence_validation"] = {"status": "PASSED"}
        return Decision(record.id, reviewed.decision, reviewed.matched_record_id, reviewed.confidence, "live_llm" if reviewed.decision == "MATCH" else "exception", reviewed.reason_codes, reviewed.evidence, reviewed.requires_human_review)
