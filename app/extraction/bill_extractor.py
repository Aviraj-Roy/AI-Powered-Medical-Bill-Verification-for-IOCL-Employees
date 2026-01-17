"""Medical Bill Extractor.

Converts *structured* OCR output into a *single*, bill-scoped document.

Non-negotiable business rules enforced:
- One PDF upload = one MongoDB document.
- Payments/receipts are NOT medical services and must be routed to `payments: []`.
- No hospital-specific logic. No LLM usage.

Design approach:
- Section-aware state tracking (page-aware ordering) to classify items deterministically.
- Exclusion-first routing for payments/receipts.
- Header locking (set-once) with field validation to prevent later-page corruption.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Payment / receipt detection (generic)
# =============================================================================
PAYMENT_PATTERNS = [
    r"\bRCPO-[A-Z0-9]+\b",
    r"\bRCPT[-/:]?[A-Z0-9]+\b",
    r"\b(UTR|RRN|TXN|TRANSACTION)\b",
    r"\b(PAYMENT|PAID|RECEIPT)\b",
    r"\b(CASH|CARD|UPI|NET\s*BANKING)\b",
]


def is_paymentish(text: str) -> bool:
    t = (text or "").upper()
    return any(re.search(p, t) for p in PAYMENT_PATTERNS)


def extract_reference(text: str) -> Optional[str]:
    if not text:
        return None
    u = text.upper()
    m = re.search(r"\bRCPO-[A-Z0-9]+\b", u)
    if m:
        return m.group(0)
    m = re.search(r"\b(UTR|RRN|TXN)\s*[:#-]?\s*([A-Z0-9]{6,})\b", u)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None


# =============================================================================
# Amount extraction
# =============================================================================
AMOUNT_PATTERNS = [
    r"₹?\s*([\d,]+\.\d{2})\s*$",
    r"₹?\s*([\d,]+)\s*$",
]


def extract_amount_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    for pat in AMOUNT_PATTERNS:
        m = re.search(pat, text.strip())
        if not m:
            continue
        s = m.group(1).replace(",", "")
        try:
            return float(s)
        except ValueError:
            continue
    return None


# =============================================================================
# Section detection (generic)
# =============================================================================
SECTION_HEADERS = {
    "medicines": ["medicine", "medicines", "drug", "drugs", "pharmacy"],
    "diagnostics_tests": [
        "diagnostic",
        "diagnostics",
        "investigation",
        "pathology",
        "laboratory",
        "lab",
        "non-lab",
        "non lab",
        "imaging",
    ],
    "radiology": ["radiology", "x-ray", "xray", "ct", "mri", "ultrasound", "usg"],
    "consultation": ["consultation", "consult", "doctor fee", "physician"],
    "hospitalization": ["hospitalisation", "hospitalization", "room", "ward", "bed", "icu", "nursing"],
    "packages": ["package", "packages", "procedure package"],
    "administrative": ["administrative", "registration", "processing", "documentation"],
    "implants_devices": ["implant", "implants", "device", "devices", "stent", "pacemaker"],
    "surgical_consumables": ["consumable", "consumables", "surgical", "gloves", "syringe", "catheter"],
}


def detect_section_header(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower().strip()

    # Avoid lines that look like amounts/items
    if len(t) > 80:
        return None
    if re.search(r"[\d,]+\.\d{2}\s*$", t):
        return None

    for section, keywords in SECTION_HEADERS.items():
        if any(k in t for k in keywords):
            return section

    return None


# =============================================================================
# Header extraction with locking
# =============================================================================
VALUE_VALIDATORS = {
    "patient_name": {
        "min_len": 3,
        "max_len": 100,
        # prevent bill numbers / MRN from landing in name
        "invalid_patterns": [r"^[A-Z]{2}\d{6,}", r"^\d{10,}$"],
        "valid_patterns": [r"[A-Za-z]{2,}"],
    },
    "patient_mrn": {"min_len": 5, "max_len": 20, "valid_patterns": [r"\d{5,}"]},
    "billing_date": {
        "min_len": 8,
        "max_len": 30,
        "valid_patterns": [r"\d{2}[-/]\d{2}[-/]\d{4}", r"\d{4}[-/]\d{2}[-/]\d{2}"],
    },
    "bill_number": {
        "min_len": 5,
        "max_len": 40,
        "valid_patterns": [r"[A-Z]{2,}\d+", r"\d+[A-Z]+\d+"],
    },
}


def _validate(field: str, value: str) -> bool:
    if not value or not value.strip():
        return False
    v = value.strip()
    rules = VALUE_VALIDATORS.get(field)
    if not rules:
        return True

    if len(v) < rules.get("min_len", 1) or len(v) > rules.get("max_len", 9999):
        return False

    for p in rules.get("invalid_patterns", []):
        if re.search(p, v, re.IGNORECASE):
            return False

    valids = rules.get("valid_patterns", [])
    if valids and not any(re.search(p, v, re.IGNORECASE) for p in valids):
        return False

    return True


@dataclass
class Candidate:
    field: str
    value: str
    score: float
    page: int


class HeaderAggregator:
    """Set-once header locking.

    First valid value wins. Later pages cannot overwrite stable header fields.
    """

    def __init__(self):
        self.best: Dict[str, Candidate] = {}

    def offer(self, cand: Candidate) -> None:
        if not _validate(cand.field, cand.value):
            return
        if cand.field not in self.best:
            self.best[cand.field] = cand
            return

        # Never overwrite once set unless existing is invalid.
        current = self.best[cand.field]
        if not _validate(current.field, current.value):
            self.best[cand.field] = cand

    def finalize(self) -> Dict[str, str]:
        return {k: v.value for k, v in self.best.items()}


LABEL_PATTERNS = {
    "patient_name": [r"patient\s*name\s*[:.]?", r"^name\s*[:.]?"],
    "patient_mrn": [r"patient\s*mrn\s*[:.]?", r"mrn\s*[:.]?", r"uhid\s*[:.]?"],
    "bill_number": [r"bill\s*no\s*[:.]?", r"bill\s*number\s*[:.]?", r"invoice\s*no\s*[:.]?"],
    "billing_date": [r"billing\s*date\s*[:.]?", r"bill\s*date\s*[:.]?"],
}


def extract_header_candidates(lines: List[Dict[str, Any]]) -> List[Candidate]:
    cands: List[Candidate] = []

    for line in lines:
        text = (line.get("text") or "").strip()
        if not text:
            continue

        page = int(line.get("page", 0) or 0)
        conf = float(line.get("confidence", 1.0) or 1.0)
        tl = text.lower()

        for field, patterns in LABEL_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, tl):
                    m = re.search(pat + r"\s*(.+)", text, re.IGNORECASE)
                    if m:
                        val = re.sub(r"^[:.]\s*", "", m.group(1).strip())
                        cands.append(Candidate(field=field, value=val, score=conf, page=page))
                    break

    return cands


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _make_id(prefix: str, parts: List[str]) -> str:
    payload = "|".join([prefix, *parts])
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


# =============================================================================
# Bill extraction
# =============================================================================
class BillExtractor:
    def extract(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        raw_text = ocr_result.get("raw_text", "") or ""
        lines: List[Dict[str, Any]] = ocr_result.get("lines") or []
        item_blocks: List[Dict[str, Any]] = ocr_result.get("item_blocks") or []

        # Legacy fallback: if only raw_text exists
        if not lines and raw_text:
            lines = [
                {"text": t.strip(), "confidence": 1.0, "box": None, "page": 0}
                for t in raw_text.split("\n")
                if t.strip()
            ]

        def y_key_line(l: Dict[str, Any]) -> float:
            box = l.get("box")
            try:
                if isinstance(box, (list, tuple)) and box:
                    return float(min(p[1] for p in box))
            except Exception:
                pass
            return 0.0

        lines_sorted = sorted(lines, key=lambda l: (int(l.get("page", 0) or 0), y_key_line(l)))

        # Header lock
        header_agg = HeaderAggregator()
        bill_number_candidates: List[str] = []

        for cand in extract_header_candidates(lines_sorted):
            header_agg.offer(cand)
            if cand.field == "bill_number" and _validate("bill_number", cand.value):
                bill_number_candidates.append(cand.value.strip())

        header_locked = header_agg.finalize()

        # Keep multiple bill numbers if present (but still one PDF -> one document)
        bill_numbers: List[str] = []
        seen = set()
        for bn in bill_number_candidates:
            bn2 = bn.strip()
            if bn2 and bn2 not in seen:
                seen.add(bn2)
                bill_numbers.append(bn2)

        primary_bill_number = header_locked.get("bill_number")
        if primary_bill_number and primary_bill_number not in seen:
            bill_numbers.insert(0, primary_bill_number)

        # Section events: PAGE-AWARE but carried across pages if no new header appears
        section_events: List[Tuple[Tuple[int, float], str]] = []  # ((page, y), section)
        for line in lines_sorted:
            sec = detect_section_header(line.get("text") or "")
            if not sec:
                continue
            page = int(line.get("page", 0) or 0)
            y = y_key_line(line)
            section_events.append(((page, y), sec))

        section_events.sort(key=lambda x: x[0])
        section_keys = [k for (k, _) in section_events]
        section_vals = [v for (_, v) in section_events]

        def section_at(page: int, y: float) -> Optional[str]:
            if not section_keys:
                return None
            # find last event with key <= (page,y)
            import bisect

            idx = bisect.bisect_right(section_keys, (page, y)) - 1
            if idx >= 0:
                return section_vals[idx]
            return None

        categorized: Dict[str, List[Dict[str, Any]]] = {
            k: []
            for k in [
                "medicines",
                "regulated_pricing_drugs",
                "surgical_consumables",
                "implants_devices",
                "diagnostics_tests",
                "radiology",
                "consultation",
                "hospitalization",
                "packages",
                "administrative",
                "other",
            ]
        }
        payments: List[Dict[str, Any]] = []

        # Prefer OCR-provided item_blocks (row-grouped). They should include page/y from OCR engine.
        if item_blocks:
            for block in item_blocks:
                text = _normalize_ws(block.get("text") or "")
                desc = _normalize_ws(block.get("description") or "") or text
                cols = block.get("columns") or []
                page = int(block.get("page", 0) or 0)
                y = float(block.get("y", 0.0) or 0.0)

                # Exclusion-first: route receipts/payments to payments[]
                if is_paymentish(text) or is_paymentish(desc):
                    amount = extract_amount_from_text(text)
                    ref = extract_reference(text)
                    pid = _make_id("payment", [ref or "", f"{amount or ''}", desc.lower(), str(page)])
                    payments.append(
                        {
                            "payment_id": pid,
                            "description": desc,
                            "amount": amount,
                            "reference": ref,
                            "page": page,
                        }
                    )
                    continue

                # Amount: prefer numeric columns
                amount: Optional[float] = None
                for c in reversed(cols):
                    amount = extract_amount_from_text(c)
                    if amount is not None:
                        break
                if amount is None:
                    amount = extract_amount_from_text(text)
                if amount is None or amount <= 0:
                    continue

                sec = section_at(page, y)
                category = sec if sec in categorized else "other"

                item_id = _make_id(
                    "item",
                    [category, f"{amount:.2f}", desc.lower(), str(page)],
                )

                categorized[category].append(
                    {
                        "item_id": item_id,
                        "description": desc,
                        "amount": amount,
                        "category": category,
                        "page": page,
                        "section_raw": sec,
                    }
                )
        else:
            # line-based fallback (less accurate)
            current_section: Optional[str] = None
            for line in lines_sorted:
                text = _normalize_ws(line.get("text") or "")
                if not text:
                    continue

                sec = detect_section_header(text)
                if sec:
                    current_section = sec
                    continue

                if is_paymentish(text):
                    amount = extract_amount_from_text(text)
                    ref = extract_reference(text)
                    pid = _make_id("payment", [ref or "", f"{amount or ''}", text.lower(), str(int(line.get("page", 0) or 0))])
                    payments.append(
                        {
                            "payment_id": pid,
                            "description": text,
                            "amount": amount,
                            "reference": ref,
                            "page": int(line.get("page", 0) or 0),
                        }
                    )
                    continue

                amt = extract_amount_from_text(text)
                if amt is None or amt <= 0:
                    continue

                category = current_section if current_section in categorized else "other"
                item_id = _make_id("item", [category, f"{amt:.2f}", text.lower(), str(int(line.get("page", 0) or 0))])
                categorized[category].append(
                    {
                        "item_id": item_id,
                        "description": text,
                        "amount": amt,
                        "category": category,
                        "page": int(line.get("page", 0) or 0),
                        "section_raw": current_section,
                    }
                )

        # Guardrail: payments must not appear in medical items
        for cat, items in categorized.items():
            for it in items:
                d = (it.get("description") or "").upper()
                if "RCPO-" in d:
                    raise AssertionError(f"Payment-like reference leaked into medical items category={cat}")

        subtotals = {
            k: round(sum(i.get("amount", 0.0) or 0.0 for i in v), 2) for k, v in categorized.items()
        }
        grand_total = round(sum(subtotals.values()), 2)

        result: Dict[str, Any] = {
            "extraction_date": datetime.now().isoformat(),
            "header": {
                "primary_bill_number": primary_bill_number,
                "bill_numbers": bill_numbers,
                "billing_date": header_locked.get("billing_date"),
            },
            "patient": {
                "name": header_locked.get("patient_name") or "UNKNOWN",
                "mrn": header_locked.get("patient_mrn"),
            },
            "items": categorized,
            "payments": payments,
            "subtotals": subtotals,
            "grand_total": grand_total,
            "raw_ocr_text": raw_text[:5000] if raw_text else None,
        }

        return result


def extract_bill_data(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """Public entry point used by the rest of the codebase."""

    return BillExtractor().extract(ocr_result)
