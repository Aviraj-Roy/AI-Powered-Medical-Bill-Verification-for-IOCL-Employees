"""Zone Detection for Medical Bill Extraction.

Detects document zones:
- Header Zone: Patient info, bill metadata (before table starts)
- Item Zone: Line items / services (between header and payment)
- Payment Zone: Receipts, payments, totals (RCPO-*, etc.)

Design principles:
- No hardcoded hospital/test names.
- Generic patterns that work across hospitals.
- Zone boundaries are Y-coordinate based per page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Zone Boundary Patterns
# =============================================================================

# Patterns that indicate TABLE/ITEMS start (end of header zone)
TABLE_START_PATTERNS = [
    r"^\s*s\.?\s*no\.?\s*$",                          # "S.No" or "S. No."
    r"^\s*sr\.?\s*no\.?\s*$",                         # "Sr. No"
    r"^\s*sl\.?\s*no\.?\s*$",                         # "Sl. No"
    r"^\s*#\s*$",                                     # "#"
    r"^\s*item\s*$",                                  # "Item"
    r"^\s*description\s*$",                           # "Description"
    r"^\s*particulars\s*$",                           # "Particulars"
    r"^\s*service\s*(name|description)?\s*$",         # "Service" / "Service Name"
    r"\bqty\b.*\brate\b.*\bamount\b",                 # "Qty Rate Amount" row
    r"\bquantity\b.*\bunit\b.*\btotal\b",             # "Quantity Unit Total"
    r"^\s*date\s+service\s+",                         # "Date Service ..."
]

# Patterns that indicate PAYMENT zone start
PAYMENT_ZONE_PATTERNS = [
    r"\bRCPO[-/]?[A-Z0-9]+\b",                        # RCPO-12345
    r"\breceipt\s*(no|number|#)?\s*[:.]?\s*\w+",     # Receipt No: XYZ
    r"\bpayment\s*(details|summary|info)",            # Payment Details
    r"\bamount\s*(paid|received)\b",                  # Amount Paid
    r"\btotal\s*paid\b",                              # Total Paid
    r"\bbalance\s*(due|to\s*pay|payable)\b",          # Balance Due
    r"\bmode\s*of\s*payment\b",                       # Mode of Payment
    r"\b(cash|card|upi|neft|rtgs|cheque)\s*payment",  # Cash Payment
    r"\bUTR\s*[:.]?\s*\d+",                           # UTR: 123456
    r"\bTXN\s*[:.]?\s*\w+",                           # TXN: ABC123
    r"\bRRN\s*[:.]?\s*\d+",                           # RRN: 123456
]

# Patterns for HEADER labels (should NOT be treated as items)
HEADER_LABEL_PATTERNS = [
    r"^\s*patient\s*(name|id|mrn|uhid)\s*[:.]?",
    r"^\s*(name|mrn|uhid)\s*[:.]?\s*$",
    r"^\s*gender\s*[:|/]?\s*(age|dob)?",
    r"^\s*age\s*[:|/]?\s*(gender|sex)?",
    r"^\s*date\s*of\s*birth\s*[:.]?",
    r"^\s*dob\s*[:.]?",
    r"^\s*address\s*[:.]?",
    r"^\s*(phone|mobile|contact)\s*(no|number)?\s*[:.]?",
    r"^\s*(bill|invoice)\s*(no|number|date)\s*[:.]?",
    r"^\s*billing\s*(date|time)\s*[:.]?",
    r"^\s*hospital\s*name\s*[:.]?",
    r"^\s*clinic\s*name\s*[:.]?",
    r"^\s*(consultant|doctor|dr\.?)\s*(name)?\s*[:.]?",
    r"^\s*visit\s*(no|number|date)\s*[:.]?",
    r"^\s*admission\s*(date|time)\s*[:.]?",
    r"^\s*discharge\s*(date|time)\s*[:.]?",
    r"^\s*gstin\s*[:.]?",
    r"^\s*reg\.?\s*(no|number)\s*[:.]?",
]

# Section headers (medical categories)
SECTION_HEADER_PATTERNS = [
    r"^\s*[-=]*\s*(medicine|medicines|pharmacy|drug|drugs)\s*[-=]*\s*$",
    r"^\s*[-=]*\s*(diagnostic|diagnostics|investigation|lab|laboratory)\s*[-=]*\s*$",
    r"^\s*[-=]*\s*(radiology|imaging|x-ray|xray|ct|mri|ultrasound)\s*[-=]*\s*$",
    r"^\s*[-=]*\s*(consultation|consult)\s*[-=]*\s*$",
    r"^\s*[-=]*\s*(hospitali[sz]ation|room|ward|bed|nursing)\s*[-=]*\s*$",
    r"^\s*[-=]*\s*(package|packages|procedure)\s*[-=]*\s*$",
    r"^\s*[-=]*\s*(consumable|consumables|surgical)\s*[-=]*\s*$",
    r"^\s*[-=]*\s*(implant|implants|device|devices)\s*[-=]*\s*$",
    r"^\s*[-=]*\s*(administrative|admin|registration)\s*[-=]*\s*$",
]


@dataclass
class ZoneBoundary:
    """Represents a zone boundary in the document."""
    page: int
    y: float
    zone_type: str  # "header_end", "payment_start"
    trigger_text: str = ""


@dataclass
class PageZones:
    """Zone boundaries for a single page."""
    page: int
    header_end_y: Optional[float] = None
    payment_start_y: Optional[float] = None
    section_headers: List[Tuple[float, str]] = field(default_factory=list)  # (y, section_name)


def _get_y(line: Dict[str, Any]) -> float:
    """Extract Y coordinate from a line dict."""
    box = line.get("box")
    if box is None:
        return 0.0
    try:
        if hasattr(box, "__iter__") and len(box) > 0:
            return float(min(p[1] for p in box))
    except (TypeError, IndexError, ValueError):
        pass
    return 0.0


def is_table_start(text: str) -> bool:
    """Check if text indicates start of item table."""
    if not text:
        return False
    t = text.lower().strip()
    return any(re.search(p, t, re.IGNORECASE) for p in TABLE_START_PATTERNS)


def is_payment_zone(text: str) -> bool:
    """Check if text indicates payment zone."""
    if not text:
        return False
    t = text.upper().strip()
    return any(re.search(p, t, re.IGNORECASE) for p in PAYMENT_ZONE_PATTERNS)


def is_header_label(text: str) -> bool:
    """Check if text is a header label (not an item)."""
    if not text:
        return False
    t = text.lower().strip()
    return any(re.search(p, t, re.IGNORECASE) for p in HEADER_LABEL_PATTERNS)


def is_section_header(text: str) -> bool:
    """Check if text is a section header."""
    if not text:
        return False
    t = text.lower().strip()
    return any(re.search(p, t, re.IGNORECASE) for p in SECTION_HEADER_PATTERNS)


def detect_zones_for_page(lines: List[Dict[str, Any]], page: int) -> PageZones:
    """Detect zone boundaries for a single page.

    Args:
        lines: OCR lines for this page (sorted by Y)
        page: Page number

    Returns:
        PageZones with detected boundaries
    """
    zones = PageZones(page=page)

    page_lines = [l for l in lines if int(l.get("page", 0) or 0) == page]
    page_lines_sorted = sorted(page_lines, key=_get_y)

    for line in page_lines_sorted:
        text = (line.get("text") or "").strip()
        y = _get_y(line)

        # Detect header end (table start)
        if zones.header_end_y is None and is_table_start(text):
            zones.header_end_y = y

        # Detect payment zone start
        if zones.payment_start_y is None and is_payment_zone(text):
            zones.payment_start_y = y

        # Detect section headers
        if is_section_header(text):
            section = _classify_section(text)
            if section:
                zones.section_headers.append((y, section))

    return zones


def detect_all_zones(lines: List[Dict[str, Any]]) -> Dict[int, PageZones]:
    """Detect zone boundaries for all pages.

    Args:
        lines: All OCR lines (will be grouped by page)

    Returns:
        Dict mapping page number to PageZones
    """
    # Group by page
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for line in lines:
        p = int(line.get("page", 0) or 0)
        pages.setdefault(p, []).append(line)

    zones: Dict[int, PageZones] = {}
    for page, page_lines in pages.items():
        zones[page] = detect_zones_for_page(page_lines, page)

    return zones


def _classify_section(text: str) -> Optional[str]:
    """Classify section header text into category."""
    t = text.lower().strip()

    section_mapping = {
        "medicines": ["medicine", "medicines", "pharmacy", "drug", "drugs"],
        "diagnostics_tests": ["diagnostic", "diagnostics", "investigation", "lab", "laboratory", "pathology"],
        "radiology": ["radiology", "imaging", "x-ray", "xray", "ct", "mri", "ultrasound", "usg"],
        "consultation": ["consultation", "consult"],
        "hospitalization": ["hospitali", "room", "ward", "bed", "nursing", "icu"],
        "packages": ["package", "packages", "procedure package"],
        "surgical_consumables": ["consumable", "consumables", "surgical"],
        "implants_devices": ["implant", "implants", "device", "devices"],
        "administrative": ["administrative", "admin", "registration"],
    }

    for section, keywords in section_mapping.items():
        if any(kw in t for kw in keywords):
            return section

    return None


def get_line_zone(
    line: Dict[str, Any],
    page_zones: Dict[int, PageZones],
) -> str:
    """Determine which zone a line belongs to.

    Args:
        line: OCR line dict
        page_zones: Zone boundaries by page

    Returns:
        Zone type: "header", "items", or "payment"
    """
    page = int(line.get("page", 0) or 0)
    y = _get_y(line)
    text = (line.get("text") or "").strip()

    # Check if explicitly a header label
    if is_header_label(text):
        return "header"

    # Check if explicitly payment
    if is_payment_zone(text):
        return "payment"

    zones = page_zones.get(page)
    if zones is None:
        return "items"  # Default to items if no zone info

    # Before header end = header zone (only on page 0)
    if page == 0 and zones.header_end_y is not None and y < zones.header_end_y:
        return "header"

    # After payment start = payment zone
    if zones.payment_start_y is not None and y >= zones.payment_start_y:
        return "payment"

    return "items"


def should_skip_as_header_label(text: str) -> bool:
    """Check if text should be skipped during item parsing because it's a header label.

    Args:
        text: Line text

    Returns:
        True if should skip (header label), False otherwise
    """
    return is_header_label(text)
