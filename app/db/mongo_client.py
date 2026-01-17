from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB client wrapper.

    Design requirements:
    - Index creation MUST NOT happen in ingestion/page processing.
      (Indexes are handled by `app/db/init_indexes.py`.)
    - Persistence MUST be bill-scoped: one upload_id -> one document.

    This class uses a singleton MongoClient to avoid reconnect storms.
    """

    _instance = None
    _lock = threading.Lock()
    _client: Optional[MongoClient] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, validate_schema: bool = False):
        if MongoDBClient._client is not None:
            self.client = MongoDBClient._client
            self.db = self.client[os.getenv("MONGO_DB_NAME", "medical_bills")]
            self.collection = self.db[os.getenv("MONGO_COLLECTION_NAME", "bills")]
            self.validate_schema = validate_schema
            return

        mongo_uri = os.getenv("MONGO_URI")
        db_name = os.getenv("MONGO_DB_NAME", "medical_bills")
        collection_name = os.getenv("MONGO_COLLECTION_NAME", "bills")

        if not mongo_uri:
            raise ValueError("MONGO_URI not found in .env")

        self.client = MongoClient(mongo_uri)
        MongoDBClient._client = self.client

        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.validate_schema = validate_schema

    def _validate_and_transform(self, bill_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_schema:
            return bill_data

        try:
            from app.db.bill_schema import BillDocument

            doc = BillDocument(**bill_data)
            return doc.to_mongo_dict()
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}. Storing raw data.")
            return bill_data

    def insert_bill(self, bill_data: Dict[str, Any]) -> str:
        """Legacy insert: creates a new document each call."""
        data_to_insert = self._validate_and_transform(bill_data)
        data_to_insert["inserted_at"] = datetime.now().isoformat()
        result = self.collection.insert_one(data_to_insert)
        return str(result.inserted_id)

    def upsert_bill(self, upload_id: str, bill_data: Dict[str, Any]) -> str:
        """Bill-scoped persistence: one upload_id -> one document.

        Uses:
        - $setOnInsert for immutable metadata
        - $set for computed fields
        - $addToSet/$each for append-only items & payments (dedupe via stable item_id/payment_id)

        NOTE: uses `_id == upload_id` to guarantee exactly one doc per upload.
        """

        data = self._validate_and_transform(bill_data)

        header = data.get("header", {}) or {}
        patient = data.get("patient", {}) or {}
        items = data.get("items", {}) or {}
        payments = data.get("payments", []) or []

        # Build $addToSet update for each category
        add_to_set: Dict[str, Any] = {"payments": {"$each": payments}}
        for category, arr in items.items():
            if not isinstance(arr, list):
                continue
            add_to_set[f"items.{category}"] = {"$each": arr}

        now = datetime.now().isoformat()

        update = {
            "$setOnInsert": {
                "_id": upload_id,
                "upload_id": upload_id,
                "created_at": now,
                "source_pdf": data.get("source_pdf"),
                "schema_version": data.get("schema_version", 1),
            },
            "$set": {
                "updated_at": now,
                "page_count": data.get("page_count"),
                "extraction_date": data.get("extraction_date"),
                "header": header,
                "patient": patient,
                "subtotals": data.get("subtotals", {}),
                "summary": data.get("summary", {}),
                "grand_total": data.get("grand_total", 0.0),
                "raw_ocr_text": data.get("raw_ocr_text"),
                "status": data.get("status", "complete"),
            },
            "$addToSet": add_to_set,
        }

        self.collection.update_one({"_id": upload_id}, update, upsert=True)
        return upload_id

    def get_bill_by_upload_id(self, upload_id: str) -> Optional[Dict[str, Any]]:
        return self.collection.find_one({"_id": upload_id})

    def get_bills_by_patient_mrn(self, mrn: str) -> List[Dict[str, Any]]:
        return list(self.collection.find({"patient.mrn": mrn}))

    def get_bills_by_patient_name(self, patient_name: str) -> List[Dict[str, Any]]:
        return list(self.collection.find({"patient.name": {"$regex": patient_name, "$options": "i"}}))

    def get_statistics(self) -> Dict[str, Any]:
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_bills": {"$sum": 1},
                    "total_revenue": {"$sum": "$grand_total"},
                    "avg_bill_amount": {"$avg": "$grand_total"},
                }
            }
        ]

        result = list(self.collection.aggregate(pipeline))
        if not result:
            return {"message": "No data available"}
        return result[0]
