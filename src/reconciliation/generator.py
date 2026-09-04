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
        invoice_id = f"INV-{index:04d}"
        records.append(FinancialRecord(invoice_id, "invoice", amount, "INR", txn_date, key, f"Invoice from {vendor}"))
        related = [invoice_id]

        mode = index % 10

        # Mode 0: Missing record (no counterpart in any source system)
        if mode == 0:
            truth["corruptions"][invoice_id] = "MISSING_RECORD"

        # Mode 1: Clean Exact Bank Match (deterministic >= 0.95)
        elif mode == 1:
            truth["corruptions"][invoice_id] = "CLEAN"
            bnk = FinancialRecord(f"BNK-{index:04d}", "bank", amount, "INR", txn_date, key, f"Credit {vendor}")
            records.append(bnk)
            related.append(bnk.id)

        # Mode 2: Amount Mismatch (fee variance -> score in [0.75, 0.95) -> AI Review)
        elif mode == 2:
            truth["corruptions"][invoice_id] = "AMOUNT_MISMATCH"
            gw_amt = round(amount * 0.975, 2)
            gw = FinancialRecord(f"GW-{index:04d}", "gateway", gw_amt, "INR", txn_date, key, f"Settlement {vendor}")
            records.append(gw)
            related.append(gw.id)

        # Mode 3: Date Mismatch (shifted 2 days -> score in [0.75, 0.95) -> AI Review)
        elif mode == 3:
            truth["corruptions"][invoice_id] = "DATE_MISMATCH"
            bnk = FinancialRecord(f"BNK-{index:04d}", "bank", amount, "INR", txn_date + timedelta(days=2), key, f"Credit {vendor}")
            records.append(bnk)
            related.append(bnk.id)

        # Mode 4: Clean Exact Ledger Match (deterministic >= 0.95)
        elif mode == 4:
            truth["corruptions"][invoice_id] = "CLEAN"
            led = FinancialRecord(f"LED-{index:04d}", "ledger", amount, "INR", txn_date, key, f"Payable {vendor}")
            records.append(led)
            related.append(led.id)

        # Mode 5: Missing Record (orphaned invoice)
        elif mode == 5:
            truth["corruptions"][invoice_id] = "MISSING_RECORD"

        # Mode 6: Gateway Fee Deduction (1.8% variance -> score in [0.75, 0.95) -> AI Review)
        elif mode == 6:
            truth["corruptions"][invoice_id] = "GATEWAY_FEE"
            gw_amt = round(amount * 0.982, 2)
            gw = FinancialRecord(f"GW-{index:04d}", "gateway", gw_amt, "INR", txn_date, key, f"Settlement {vendor}")
            records.append(gw)
            related.append(gw.id)

        # Mode 7: Date Mismatch (shifted 3 days -> score in [0.75, 0.95) -> AI Review)
        elif mode == 7:
            truth["corruptions"][invoice_id] = "DATE_MISMATCH"
            gw = FinancialRecord(f"GW-{index:04d}", "gateway", amount, "INR", txn_date + timedelta(days=3), key, f"Settlement {vendor}")
            records.append(gw)
            related.append(gw.id)

        # Mode 8: Clean Exact Gateway Match (deterministic >= 0.95)
        elif mode == 8:
            truth["corruptions"][invoice_id] = "CLEAN"
            gw = FinancialRecord(f"GW-{index:04d}", "gateway", amount, "INR", txn_date, key, f"Settlement {vendor}")
            records.append(gw)
            related.append(gw.id)

        # Mode 9: Duplicate Bank Records (identical candidates -> multiple candidates)
        elif mode == 9:
            truth["corruptions"][invoice_id] = "DUPLICATE"
            b1 = FinancialRecord(f"BNK-{index:04d}", "bank", amount, "INR", txn_date, key, f"Credit {vendor}")
            b2 = FinancialRecord(f"BNK-DUP-{index:04d}", "bank", amount, "INR", txn_date, key, f"Credit {vendor}")
            records.extend([b1, b2])
            related.extend([b1.id, b2.id])

        truth["relationships"][invoice_id] = related

    return records, truth


def save_ground_truth(ground_truth: dict[str, Any], path: Path) -> None:
    """Persist generated truth separately from operating reconciliation data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
