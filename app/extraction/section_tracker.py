"""Stateful Section Tracking for Medical Bill Extraction.

Maintains section context as lines are processed:
- Detects section headers (Diagnostics, Radiology, Consultation, etc.)
- Persists section context across pages until new header found
- Assigns items to categories based on active section

Design principles:
- Section context carries forward to next page if no new header.
- Explicit section header resets context.
- Items without section context go to "other".
- No hardcoded hospital/test names.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Section Keywords (Generic - No Hospital-Specific Terms)
# =============================================================================
SECTION_KEYWORDS = {
    "medicines": [
        "medicine", "medicines", "pharmacy", "drug", "drugs",
        "medication", "medications", "pharma",
    ],
    "diagnostics_tests": [
        "diagnostic", "diagnostics", "investigation", "investigations",
        "pathology", "laboratory", "lab", "lab services",
        "non-lab", "non lab", "test", "tests",
    ],
    "radiology": [
        "radiology", "imaging", "x-ray", "xray", "ct scan", "ct",
        "mri", "ultrasound", "usg", "sonography", "scan",
    ],
    "consultation": [
        "consultation", "consult", "consultations",
        "doctor fee", "physician", "specialist",
    ],
    "hospitalization": [
        "hospitalisation", "hospitalization", "room", "ward", "bed",
        "icu", "nursing", "accommodation", "stay", "room charges",
    ],
    "packages": [
        "package", "packages", "procedure package", "health package",
        "checkup package",
    ],
    "surgical_consumables": [
        "consumable", "consumables", "surgical consumable",
        "surgical supplies", "disposable", "disposables",
    ],
    "implants_devices": [
        "implant", "implants", "device", "devices",
        "stent", "pacemaker", "prosthesis",
    ],
    "administrative": [
        "administrative", "admin", "registration", "processing",
        "documentation", "admission", "discharge",
    ],
    "regulated_pricing_drugs": [
        "regulated pricing", "dpco", "nlem", "price regulated",
    ],
}

# Valid categories for item classification
VALID_CATEGORIES = list(SECTION_KEYWORDS.keys()) + ["other"]


@dataclass
class SectionEvent:
    """Represents a section header detected in the document."""
    page: int
    y: float
    section: str
    text: str = ""


@dataclass
class SectionTracker:
    """Stateful tracker for document sections.

    Maintains section context as lines are processed.
    Section persists across pages until a new section header is found.
    """

    events: List[SectionEvent] = field(default_factory=list)
    _sorted: bool = False
    _keys: List[Tuple[int, float]] = field(default_factory=list)
    _values: List[str] = field(default_factory=list)

    def add_event(self, page: int, y: float, section: str, text: str = "") -> None:
        """Register a section header event.

        Args:
            page: Page number (0-indexed)
            y: Y coordinate of the section header
            section: Category name (e.g., "diagnostics_tests")
            text: Original text of the section header
        """
        self.events.append(SectionEvent(page=page, y=y, section=section, text=text))
        self._sorted = False

    def _ensure_sorted(self) -> None:
        """Sort events by (page, y) for binary search."""
        if self._sorted:
            return

        self.events.sort(key=lambda e: (e.page, e.y))
        self._keys = [(e.page, e.y) for e in self.events]
        self._values = [e.section for e in self.events]
        self._sorted = True

    def get_section_at(self, page: int, y: float) -> Optional[str]:
        """Get the active section at a given position.

        Finds the most recent section header before (page, y).
        Section context persists across pages.

        Args:
            page: Page number
            y: Y coordinate

        Returns:
            Section name or None if no section context
        """
        self._ensure_sorted()

        if not self._keys:
            return None

        # Find the last event with key <= (page, y)
        idx = bisect.bisect_right(self._keys, (page, y)) - 1

        if idx >= 0:
            return self._values[idx]

        return None

    def classify_position(self, page: int, y: float) -> str:
        """Get category for a position, defaulting to 'other'.

        Args:
            page: Page number
            y: Y coordinate

        Returns:
            Category name (never None)
        """
        section = self.get_section_at(page, y)
        return section if section in VALID_CATEGORIES else "other"


def detect_section_header(text: str) -> Optional[str]:
    """Detect if text is a section header and return the category.

    Args:
        text: Line text to check

    Returns:
        Category name if section header, None otherwise
    """
    if not text:
        return None

    t = text.lower().strip()

    # Skip if too long (likely not a section header)
    if len(t) > 60:
        return None

    # Skip if looks like an item (has amount at end)
    if re.search(r"[\d,]+\.\d{2}\s*$", t):
        return None

    # Check each category's keywords
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            # Match keyword at word boundary
            # Allow section headers like "--- DIAGNOSTICS ---" or "DIAGNOSTICS:"
            pattern = rf"(^|\s|[-=:])({re.escape(kw)})(\s|[-=:]|$)"
            if re.search(pattern, t, re.IGNORECASE):
                return section

    return None


def build_section_tracker(lines: List[Dict[str, Any]]) -> SectionTracker:
    """Build a SectionTracker from OCR lines.

    Args:
        lines: OCR lines with 'text', 'page', and 'box' fields

    Returns:
        Populated SectionTracker
    """
    tracker = SectionTracker()

    for line in lines:
        text = (line.get("text") or "").strip()
        page = int(line.get("page", 0) or 0)
        y = _get_y(line)

        section = detect_section_header(text)
        if section:
            tracker.add_event(page=page, y=y, section=section, text=text)

    return tracker


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


def classify_item_by_description(description: str) -> Optional[str]:
    """Attempt to classify an item by its description text.

    This is a fallback when no section context is available.
    Uses keyword matching similar to section detection.

    Args:
        description: Item description

    Returns:
        Category name or None if cannot classify
    """
    if not description:
        return None

    t = description.lower().strip()

    # Check keywords with lower threshold (item descriptions are usually longer)
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return section

    # Additional item-level patterns
    item_patterns = {
        "medicines": [
            r"\d+\s*mg\b",       # Dosage: 500mg
            r"\d+\s*ml\b",       # Volume: 100ml
            r"\btablet\b",
            r"\bcapsule\b",
            r"\bsyrup\b",
            r"\binjection\b",
        ],
        "diagnostics_tests": [
            r"\btest\b",
            r"\bprofile\b",
            r"\bpanel\b",
            r"\bculture\b",
            r"\bhemoglobin\b",
            r"\bcbc\b",
            r"\blft\b",
            r"\bkft\b",
            r"\brft\b",
        ],
        "radiology": [
            r"\bx[-\s]?ray\b",
            r"\bct\s*scan\b",
            r"\bmri\b",
            r"\bultrasound\b",
            r"\busg\b",
            r"\becho\b",
        ],
        "consultation": [
            r"\bconsult\b",
            r"\bvisit\b",
            r"\bopinion\b",
        ],
        "hospitalization": [
            r"\broom\s*charge\b",
            r"\bbed\s*charge\b",
            r"\bward\b",
            r"\bicu\b",
            r"\bnursing\b",
        ],
    }

    for section, patterns in item_patterns.items():
        for pattern in patterns:
            if re.search(pattern, t, re.IGNORECASE):
                return section

    return None


def get_category_for_item(
    description: str,
    page: int,
    y: float,
    tracker: SectionTracker,
) -> str:
    """Determine category for an item using section context and description.

    Priority:
    1. Active section context from tracker
    2. Keyword classification from description
    3. Default to "other"

    Args:
        description: Item description
        page: Page number
        y: Y coordinate
        tracker: SectionTracker with section events

    Returns:
        Category name
    """
    # First try section context
    section = tracker.get_section_at(page, y)
    if section and section in VALID_CATEGORIES:
        return section

    # Fallback to description-based classification
    classified = classify_item_by_description(description)
    if classified:
        return classified

    return "other"
