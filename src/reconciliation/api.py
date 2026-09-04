"""FastAPI adapter for the AI Finance Controller prototype."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .service import ReconciliationService

app = FastAPI(title="AI Finance Controller", version="0.2.0")
service = ReconciliationService()


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the finance operations dashboard."""
    return FileResponse("dashboard/finance_controller.html")


@app.post("/generate-data")
def generate_data(seed: int = 42, invoice_count: int = 120) -> dict:
    """Generate reproducible synthetic financial source data."""
    return service.generate_data(seed, invoice_count)


@app.post("/reconcile")
def reconcile() -> dict:
    """Process generated data through the reconciliation pipeline."""
    try:
        return service.reconcile()
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/status")
def status() -> dict:
    """Return current reconciliation progress and metrics."""
    return service.status()


@app.get("/records")
def records() -> list[dict]:
    """List all loaded source records."""
    return [item.to_dict() for item in service.records]


@app.get("/records/{record_id}")
def record(record_id: str) -> dict:
    """Get a record and reconciliation decision."""
    result = service.record(record_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return result


@app.get("/exceptions")
def exceptions(reason: str | None = None) -> list[dict]:
    """List exceptions, optionally filtered by reason code."""
    cases = (
        service.exceptions
        if reason is None
        else [item for item in service.exceptions if reason in item.reason_codes]
    )
    return [item.to_dict() for item in cases]


@app.get("/metrics")
def metrics() -> dict:
    """Return independent ground-truth evaluation metrics."""
    return service.metrics


@app.get("/audit/{record_id}")
def audit(record_id: str) -> dict:
    """Return the complete auditable evidence behind a decision."""
    result = service.audits.get(record_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Audit entry for '{record_id}' not found")
    return result.to_dict()


@app.get("/activity")
def activity() -> dict:
    """Return pipeline stages and live activity log entries."""
    return {
        "pipeline_stages": service.pipeline_stages,
        "activity_log": service.activity_log,
    }
