"""Exposed tools and functions for the AI reconciliation review agent."""

from __future__ import annotations

from typing import Any, Sequence

from .models import Candidate, FinancialRecord


class ReconciliationAgentTools:
    """Encapsulated tool suite for the AI reconciliation reviewer.
    
    Ensures that the model can only query, compare, and reason over real records
    present in the current universe, and cannot invent data.
    """

    def __init__(self, records: Sequence[FinancialRecord], candidates_map: dict[str, list[Candidate]] | None = None) -> None:
        self._records_by_id = {r.id: r for r in records}
        self._candidates_map = candidates_map or {}
        self.call_log: list[dict[str, Any]] = []

    def _record_call(self, tool_name: str, arguments: dict[str, Any], result: Any) -> Any:
        self.call_log.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
        })
        return result

    def get_transaction(self, record_id: str) -> dict[str, Any] | None:
        """Fetch a normalized transaction record by its exact identifier."""
        rec = self._records_by_id.get(record_id)
        result = rec.to_dict() if rec else None
        return self._record_call("get_transaction", {"record_id": record_id}, result)

    def search_transactions(self, query: dict[str, Any] | str) -> list[dict[str, Any]]:
        """Search transactions matching specific fields (e.g. reference, vendor text)."""
        matches: list[dict[str, Any]] = []
        if isinstance(query, str):
            q = query.casefold()
            for r in self._records_by_id.values():
                if q in r.id.casefold() or q in r.reference.casefold() or q in r.description.casefold():
                    matches.append(r.to_dict())
        elif isinstance(query, dict):
            for r in self._records_by_id.values():
                d = r.to_dict()
                match = True
                for k, v in query.items():
                    if k in d:
                        if isinstance(v, (int, float)):
                            if abs(d[k] - v) > 0.01:
                                match = False
                                break
                        elif str(v).casefold() not in str(d[k]).casefold():
                            match = False
                            break
                    else:
                        match = False
                        break
                if match:
                    matches.append(d)
        return self._record_call("search_transactions", {"query": query}, matches[:10])

    def get_candidates(self, record_id: str) -> list[dict[str, Any]]:
        """Fetch scored candidate matches supplied for this specific source record."""
        candidates = self._candidates_map.get(record_id, [])
        result = [c.to_dict() for c in candidates]
        return self._record_call("get_candidates", {"record_id": record_id}, result)

    def compare_records(self, record_a_id: str, record_b_id: str) -> dict[str, Any]:
        """Perform a structured side-by-side comparison between two records."""
        rec_a = self._records_by_id.get(record_a_id)
        rec_b = self._records_by_id.get(record_b_id)
        if not rec_a or not rec_b:
            res = {"error": f"One or both records not found: {record_a_id}, {record_b_id}"}
            return self._record_call("compare_records", {"record_a_id": record_a_id, "record_b_id": record_b_id}, res)

        diff = {
            "record_a": rec_a.to_dict(),
            "record_b": rec_b.to_dict(),
            "amount_diff": round(abs(rec_a.amount - rec_b.amount), 2),
            "amount_ratio": round(min(rec_a.amount, rec_b.amount) / max(rec_a.amount, rec_b.amount, 1.0), 4),
            "days_diff": abs((rec_a.transaction_date - rec_b.transaction_date).days),
            "reference_match": rec_a.reference.strip().casefold() == rec_b.reference.strip().casefold(),
            "currency_match": rec_a.currency == rec_b.currency,
        }
        return self._record_call("compare_records", {"record_a_id": record_a_id, "record_b_id": record_b_id}, diff)

    def calculate_difference(self, amount_a: float, amount_b: float) -> dict[str, float]:
        """Calculate the absolute, relative, and percentage variance between two amounts."""
        abs_diff = round(abs(amount_a - amount_b), 2)
        base = max(abs(amount_a), abs(amount_b), 1.0)
        pct_diff = round((abs_diff / base) * 100, 3)
        res = {
            "amount_a": amount_a,
            "amount_b": amount_b,
            "absolute_difference": abs_diff,
            "percentage_difference": pct_diff,
        }
        return self._record_call("calculate_difference", {"amount_a": amount_a, "amount_b": amount_b}, res)

    def mark_match(
        self,
        source_id: str,
        target_id: str,
        confidence: float,
        evidence: list[str],
        reason_codes: list[str],
    ) -> dict[str, Any]:
        """Record intent to match source_id with target_id."""
        res = {
            "decision": "MATCH",
            "source_record_id": source_id,
            "matched_record_id": target_id,
            "confidence": round(confidence, 4),
            "evidence": evidence,
            "reason_codes": reason_codes,
            "requires_human_review": False,
        }
        return self._record_call("mark_match", {"source_id": source_id, "target_id": target_id, "confidence": confidence}, res)

    def create_exception(
        self,
        source_id: str,
        reason_codes: list[str],
        confidence: float,
        evidence: list[str],
        recommended_action: str,
    ) -> dict[str, Any]:
        """Record intent to route source_id to exceptions."""
        res = {
            "decision": "EXCEPTION",
            "source_record_id": source_id,
            "matched_record_id": None,
            "confidence": round(confidence, 4),
            "evidence": evidence,
            "reason_codes": reason_codes,
            "recommended_action": recommended_action,
            "requires_human_review": True,
        }
        return self._record_call("create_exception", {"source_id": source_id, "reason_codes": reason_codes, "confidence": confidence}, res)

    def get_reconciliation_status(self) -> dict[str, Any]:
        """Return total record universe counts available to the agent."""
        res = {
            "total_records_in_memory": len(self._records_by_id),
            "total_sources": len({r.source for r in self._records_by_id.values()}),
            "tool_calls_executed": len(self.call_log),
        }
        return self._record_call("get_reconciliation_status", {}, res)

    def generate_report(self) -> dict[str, Any]:
        """Generate a trace report of all tool calls made during the current review session."""
        res = {
            "total_calls": len(self.call_log),
            "calls": list(self.call_log),
        }
        return self._record_call("generate_report", {}, res)
