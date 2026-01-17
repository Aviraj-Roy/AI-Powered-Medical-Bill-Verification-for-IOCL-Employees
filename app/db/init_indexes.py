"""MongoDB index initialization.

IMPORTANT:
- Do NOT call this from per-upload/per-page processing.
- Run it once at app startup / deploy time.

This module creates indexes in an idempotent way and validates index specs.
If an existing index conflicts with the desired spec, it raises an error
(instead of spamming runtime warnings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pymongo import ASCENDING

from app.db.mongo_client import MongoDBClient


@dataclass(frozen=True)
class IndexSpec:
    name: str
    keys: List[Tuple[str, int]]
    unique: bool = False
    sparse: bool = False
    partialFilterExpression: Optional[Dict[str, Any]] = None


def _keys_list(index_info: Dict[str, Any]) -> List[Tuple[str, int]]:
    # `key` is usually a SON/dict preserving order
    key = index_info.get("key")
    if hasattr(key, "items"):
        return list(key.items())
    return list(key) if isinstance(key, list) else []


def _index_matches(existing: Dict[str, Any], desired: IndexSpec) -> bool:
    if _keys_list(existing) != desired.keys:
        return False

    # Only compare options we explicitly care about
    if bool(existing.get("unique", False)) != bool(desired.unique):
        return False
    if bool(existing.get("sparse", False)) != bool(desired.sparse):
        return False

    # partialFilterExpression must match exactly if specified
    if desired.partialFilterExpression is not None:
        if existing.get("partialFilterExpression") != desired.partialFilterExpression:
            return False

    return True


def ensure_indexes() -> None:
    db = MongoDBClient()
    col = db.collection

    desired: List[IndexSpec] = [
        IndexSpec(
            name="idx_patient_mrn",
            keys=[("patient.mrn", ASCENDING)],
            sparse=True,
        ),
        IndexSpec(
            name="idx_patient_name",
            keys=[("patient.name", ASCENDING)],
            sparse=True,
        ),
        IndexSpec(
            name="idx_primary_bill_number",
            keys=[("header.primary_bill_number", ASCENDING)],
            sparse=True,
        ),
        IndexSpec(
            name="idx_bill_numbers",
            keys=[("header.bill_numbers", ASCENDING)],
            sparse=True,
        ),
        IndexSpec(
            name="idx_source_pdf",
            keys=[("source_pdf", ASCENDING)],
            sparse=True,
        ),
        IndexSpec(
            name="idx_created_at",
            keys=[("created_at", ASCENDING)],
            sparse=True,
        ),
    ]

    existing = list(col.list_indexes())

    for spec in desired:
        # Satisfy index if ANY existing index matches spec (even if name differs)
        matches = [ix for ix in existing if _index_matches(ix, spec)]
        if matches:
            continue

        # If same name exists but differs, that's a migration problem: fail fast.
        by_name = [ix for ix in existing if ix.get("name") == spec.name]
        if by_name:
            raise RuntimeError(
                f"Index name '{spec.name}' exists but does not match desired spec. Existing={by_name[0]} Desired={spec}"
            )

        col.create_index(
            spec.keys,
            name=spec.name,
            unique=spec.unique,
            sparse=spec.sparse,
            partialFilterExpression=spec.partialFilterExpression,
            background=True,
        )


if __name__ == "__main__":
    ensure_indexes()
    print("Indexes ensured.")
