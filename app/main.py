from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List

from app.db.mongo_client import MongoDBClient
from app.extraction.bill_extractor import extract_bill_data
from app.extraction.numeric_guards import (
    MAX_GRAND_TOTAL,
    MAX_LINE_ITEM_AMOUNT,
    validate_grand_total,
)
from app.ingestion.pdf_loader import pdf_to_images
from app.ocr.image_preprocessor import preprocess_image
from app.ocr.paddle_engine import run_ocr

logger = logging.getLogger(__name__)


# =============================================================================
# Post-Extraction Validation
# =============================================================================
class ExtractionValidationError(Exception):
    """Raised when extraction produces invalid/suspect data."""
    pass


def validate_extraction(bill_data: Dict[str, Any]) -> List[str]:
    """Validate extracted bill data against sanity checks.

    IMPORTANT: This function must be called ONCE per upload_id, AFTER full
    aggregation is complete. It validates the final merged object that will
    be persisted to MongoDB.

    Args:
        bill_data: Fully aggregated bill document (not intermediate page results)

    Returns:
        List of warning messages (empty if all valid), deduplicated

    Raises:
        ExtractionValidationError: If critical validation fails
    """
    warnings: List[str] = []

    # Validate grand total
    grand_total = bill_data.get("grand_total", 0.0) or 0.0
    is_valid, reason = validate_grand_total(grand_total)
    if not is_valid:
        warnings.append(f"Grand total validation failed: {reason} (total={grand_total})")
        # Cap it but don't fail
        bill_data["grand_total"] = min(grand_total, MAX_GRAND_TOTAL)

    # Validate individual item amounts
    items = bill_data.get("items", {}) or {}
    for category, item_list in items.items():
        for item in item_list:
            amount = item.get("amount", 0.0) or 0.0
            if amount > MAX_LINE_ITEM_AMOUNT:
                warnings.append(
                    f"Item amount exceeds cap: {item.get('description', '')[:30]} = {amount}"
                )

    # Validate no payment descriptions in items
    payments = bill_data.get("payments", []) or []
    payment_refs = set()
    for p in payments:
        ref = p.get("reference")
        if ref:
            payment_refs.add(ref.upper())

    for category, item_list in items.items():
        for item in item_list:
            desc = (item.get("description") or "").upper()
            for ref in payment_refs:
                if ref in desc:
                    raise ExtractionValidationError(
                        f"Payment reference '{ref}' found in medical item: {desc[:50]}"
                    )

    # Validate patient info exists (check FINAL aggregated state)
    patient = bill_data.get("patient", {}) or {}
    patient_name = (patient.get("name") or "").strip()
    patient_mrn = (patient.get("mrn") or "").strip()
    # Only warn if BOTH name is missing/unknown AND mrn is empty
    has_valid_name = patient_name and patient_name.upper() != "UNKNOWN"
    has_valid_mrn = bool(patient_mrn)
    if not has_valid_name and not has_valid_mrn:
        warnings.append("Patient identification missing (no name or MRN)")

    # Validate header exists (check FINAL aggregated state)
    header = bill_data.get("header", {}) or {}
    primary_bill = (header.get("primary_bill_number") or "").strip()
    bill_numbers = header.get("bill_numbers") or []
    # Filter out empty strings from bill_numbers list
    valid_bill_numbers = [bn for bn in bill_numbers if (bn or "").strip()]
    if not primary_bill and not valid_bill_numbers:
        warnings.append("No bill number extracted")

    # Deduplicate warnings (in case of any duplicate sources)
    return list(dict.fromkeys(warnings))


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def process_bill(pdf_path: str, upload_id: str | None = None) -> str:
    """Process a medical bill PDF and persist ONE MongoDB document.

    Business rules enforced:
    - One PDF upload == one MongoDB document, even if multiple pages/bill numbers.
    - Payments are NOT medical services.
    - No hardcoded hospital/test names.

    Args:
        pdf_path: Path to the PDF file
        upload_id: Optional stable upload ID (generated if not provided)

    Returns:
        The upload_id used for storage

    Raises:
        ExtractionValidationError: If critical validation fails
    """
    upload_id = upload_id or uuid.uuid4().hex

    # 1) Convert ALL pages to images
    image_paths = pdf_to_images(pdf_path)
    logger.info(f"Converted {len(image_paths)} pages from {pdf_path}")

    # 2) Preprocess ALL images
    processed_paths = [preprocess_image(p) for p in image_paths]

    # 3) OCR ALL pages together (page-aware)
    ocr_result = run_ocr(processed_paths)
    logger.info(f"OCR completed: {len(ocr_result.get('lines', []))} lines extracted")

    # 4) Extract bill-scoped structured data (three-stage pipeline)
    #    This returns fully aggregated data across ALL pages
    bill_data = extract_bill_data(ocr_result)

    # 5) Add immutable metadata BEFORE validation
    #    (so validation sees the complete final state)
    bill_data["upload_id"] = upload_id
    bill_data["source_pdf"] = os.path.basename(pdf_path)
    bill_data["page_count"] = len(image_paths)
    bill_data.setdefault("schema_version", 1)

    # 6) Post-extraction validation on FINAL aggregated object
    #    This runs exactly ONCE per upload_id after full aggregation
    warnings = validate_extraction(bill_data)
    for w in warnings:
        logger.warning(f"Extraction warning: {w}")

    # 7) Log extraction summary
    total_items = sum(len(v) for v in bill_data.get("items", {}).values())
    total_payments = len(bill_data.get("payments", []))
    logger.info(
        f"Extraction complete: {total_items} items, {total_payments} payments, "
        f"grand_total={bill_data.get('grand_total', 0)}"
    )

    # 8) Single bill-scoped upsert
    db = MongoDBClient(validate_schema=False)
    db.upsert_bill(upload_id, bill_data)
    logger.info(f"Stored bill with upload_id: {upload_id}")

    return upload_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bill_id = process_bill("T_Bill.pdf")
    print(f"Stored bill with upload_id: {bill_id}")
