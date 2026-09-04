"""Reproducible synthetic multi-source finance data generation."""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .models import FinancialRecord


def generate_synthetic_data(seed: int = 42, invoice_count: int = 120) -> tuple[list[FinancialRecord], dict[str, Any]]:
    """Create linked synthetic invoices, ledger, gateway, and bank records.

    Parameters
    ----------
    seed : int, default 42
        Fixed seed controlling every generated value and corruption.
    invoice_count : int, default 120
        Number of canonical invoices. The returned records total substantially more.

    Returns
    -------
    tuple[list[FinancialRecord], dict[str, Any]]
        Records and hidden relationship/corruption truth. This generator is only for
        demonstration and must not be used as empirical market data.
    """
    if invoice_count < 100:
        raise ValueError("invoice_count must be at least 100 for the controller demo.")
    rng = random.Random(seed)
    base = date(2026, 1, 2)
    records: list[FinancialRecord] = []
    truth: dict[str, Any] = {"seed": seed, "relationships": {}, "corruptions": {}}
    vendors = ["Northstar Services", "Cedar Retail", "Aster Logistics", "Kite Labs"]
    for index in range(1, invoice_count + 1):
        key = f"TXN-{index:04d}"
        amount = round(rng.uniform(950, 85000), 2)
        txn_date = base + timedelta(days=index % 55)
        vendor = vendors[index % len(vendors)]
        invoice_id, ledger_id = f"INV-{index:04d}", f"LED-{index:04d}"
        records.extend([
            FinancialRecord(invoice_id, "invoice", amount, "INR", txn_date, key, f"Invoice from {vendor}"),
            FinancialRecord(ledger_id, "ledger", amount, "INR", txn_date, key, f"Payable {vendor}"),
        ])
        related = [invoice_id, ledger_id]
        corruption = "CLEAN"
        # Deterministic, deliberately varied quality defects across payment sources.
        mode = index % 10
        if mode == 0:
            corruption = "MISSING_RECORD"
        else:
            gateway_id = f"GW-{index:04d}"
            gateway_amount = amount if mode != 1 else round(amount * 0.982, 2)
            gateway_date = txn_date + timedelta(days=2 if mode == 2 else 0)
            gateway_ref = key if mode != 3 else f"PAY-{index:04d}"
            gateway_desc = f"Settlement {vendor}" if mode != 4 else f"Settle {vendor.replace(' ', '')}"
            records.append(FinancialRecord(gateway_id, "gateway", gateway_amount, "INR", gateway_date, gateway_ref, gateway_desc))
            related.append(gateway_id)
            if mode in {1, 2, 3, 4}:
                corruption = {1: "AMOUNT_MISMATCH", 2: "DATE_MISMATCH", 3: "REFERENCE_MISMATCH", 4: "DESCRIPTION_MISMATCH"}[mode]
        if mode == 5:
            corruption = "MISSING_RECORD"
        else:
            bank_id = f"BNK-{index:04d}"
            bank_amount = amount if mode != 6 else round(amount * 0.99, 2)
            bank_date = txn_date + timedelta(days=3 if mode == 7 else 0)
            bank_ref = key if mode != 8 else f"NEFT{index:04d}"
            records.append(FinancialRecord(bank_id, "bank", bank_amount, "INR", bank_date, bank_ref, f"Credit {vendor}"))
            related.append(bank_id)
            if mode in {6, 7, 8}:
                corruption = {6: "GATEWAY_FEE", 7: "DATE_MISMATCH", 8: "REFERENCE_MISMATCH"}[mode]
        if mode == 9:
            duplicate = FinancialRecord(f"BNK-DUP-{index:04d}", "bank", amount, "INR", txn_date, key, f"Credit {vendor}")
            records.append(duplicate)
            related.append(duplicate.id)
            corruption = "DUPLICATE"
        truth["relationships"][invoice_id] = related
        truth["corruptions"][invoice_id] = corruption
    return records, truth


def save_ground_truth(ground_truth: dict[str, Any], path: Path) -> None:
    """Persist generated truth separately from operating reconciliation data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
