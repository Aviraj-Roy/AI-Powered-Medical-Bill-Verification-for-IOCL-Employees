from __future__ import annotations

import os
import uuid

from app.db.mongo_client import MongoDBClient
from app.extraction.bill_extractor import extract_bill_data
from app.ingestion.pdf_loader import pdf_to_images
from app.ocr.image_preprocessor import preprocess_image
from app.ocr.paddle_engine import run_ocr


def process_bill(pdf_path: str, upload_id: str | None = None) -> str:
    """Process a medical bill PDF and persist ONE MongoDB document.

    Business rule: one PDF upload == one MongoDB document, even if multiple pages / bill numbers.
    """

    upload_id = upload_id or uuid.uuid4().hex

    # 1) Convert ALL pages to images
    image_paths = pdf_to_images(pdf_path)

    # 2) Preprocess ALL images
    processed_paths = [preprocess_image(p) for p in image_paths]

    # 3) OCR ALL pages together (page-aware)
    ocr_result = run_ocr(processed_paths)

    # 4) Extract bill-scoped structured data
    bill_data = extract_bill_data(ocr_result)

    # 5) Add immutable metadata
    bill_data["upload_id"] = upload_id
    bill_data["source_pdf"] = os.path.basename(pdf_path)
    bill_data["page_count"] = len(image_paths)
    bill_data.setdefault("schema_version", 1)

    # 6) Single bill-scoped upsert
    db = MongoDBClient(validate_schema=False)
    db.upsert_bill(upload_id, bill_data)

    return upload_id


if __name__ == "__main__":
    bill_id = process_bill("M_Bill.pdf")
    print(f"Stored bill with upload_id: {bill_id}")
