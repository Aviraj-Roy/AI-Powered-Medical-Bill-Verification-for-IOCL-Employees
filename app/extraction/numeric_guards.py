"""Numeric Guardrails for Medical Bill Extraction.

Prevents false positives from phone numbers, MRNs, dates, and other
non-monetary numeric sequences being interpreted as amounts.

Design principles:
- Reject suspect patterns BEFORE they pollute downstream logic.
- Sanity caps prevent absurd totals (e.g., 2e13).
- No hardcoded hospital/test names.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# =============================================================================
# Sanity Caps
# =============================================================================
MAX_LINE_ITEM_AMOUNT = 1e7       # ₹1 crore per line item
MAX_GRAND_TOTAL = 1e8            # ₹10 crore per bill
MIN_VALID_AMOUNT = 0.01          # Minimum valid amount


# =============================================================================
# Suspect Numeric Patterns
# =============================================================================
SUSPECT_PATTERNS = [
    # Phone numbers (Indian): 10 digits starting with 6-9
    (r"^[6-9]\d{9}$", "phone_number"),

    # Phone with country code
    (r"^\+?91[-\s]?[6-9]\d{9}$", "phone_number"),

    # MRN / UHID: 12+ digit sequences
    (r"^\d{12,}$", "mrn_uhid"),

    # MRN with prefix: XX1234567890
    (r"^[A-Z]{1,3}\d{10,}$", "mrn_uhid"),

    # Dates: DD/MM/YYYY or DD-MM-YYYY
    (r"^\d{2}[-/]\d{2}[-/]\d{4}$", "date"),

    # Dates: YYYY-MM-DD or YYYY/MM/DD
    (r"^\d{4}[-/]\d{2}[-/]\d{2}$", "date"),

    # Reference IDs: RCPO-, TXN-, UTR-, RRN-
    (r"^(RCPO|TXN|UTR|RRN|REF)[-/]?\d+", "reference_id"),

    # Bill/Invoice numbers: BL123456, INV-123456
    (r"^(BL|INV|BILL)[-/]?\d{6,}$", "bill_number"),

    # Timestamps: HH:MM:SS
    (r"^\d{2}:\d{2}(:\d{2})?$", "timestamp"),

    # PIN codes (6 digits)
    (r"^\d{6}$", "pin_code"),

    # GST numbers: 22AAAAA0000A1Z5
    (r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z\d]$", "gstin"),
]


def classify_suspect_numeric(text: str) -> Optional[str]:
    """Classify a numeric string as suspect type or None if valid.

    Args:
        text: The text to classify

    Returns:
        Suspect type string (e.g., "phone_number", "mrn_uhid") or None if valid
    """
    if not text:
        return None

    cleaned = text.strip().upper().replace(",", "").replace(" ", "")

    for pattern, suspect_type in SUSPECT_PATTERNS:
        if re.match(pattern, cleaned, re.IGNORECASE):
            return suspect_type

    return None


def is_suspect_numeric(text: str) -> bool:
    """Check if a numeric string looks like a non-monetary value.

    Args:
        text: The text to check

    Returns:
        True if the text looks like phone/MRN/date/reference, False otherwise
    """
    return classify_suspect_numeric(text) is not None


# =============================================================================
# Amount Validation
# =============================================================================
def extract_numeric_value(text: str) -> Optional[float]:
    """Extract a numeric value from text, handling currency symbols and commas.

    Args:
        text: Text potentially containing a numeric value

    Returns:
        Float value or None if no valid number found
    """
    if not text:
        return None

    # Remove currency symbols and whitespace
    cleaned = text.strip()
    cleaned = re.sub(r"[₹$€£¥]", "", cleaned)
    cleaned = cleaned.strip()

    # Try to extract amount pattern
    patterns = [
        r"([+-]?[\d,]+\.\d{1,2})$",     # With decimals: 1,234.56
        r"([+-]?[\d,]+)$",              # Without decimals: 1,234
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            num_str = match.group(1).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                continue

    return None


def validate_amount(
    amount: Optional[float],
    row_has_description: bool = True,
    source_text: str = "",
) -> Tuple[bool, Optional[str]]:
    """Validate an extracted amount against sanity checks.

    Args:
        amount: The extracted amount
        row_has_description: Whether the row has descriptive text
        source_text: Original text the amount was extracted from

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    if amount is None:
        return False, "null_amount"

    # Check sanity caps
    if amount < MIN_VALID_AMOUNT:
        return False, "below_minimum"

    if amount > MAX_LINE_ITEM_AMOUNT:
        return False, f"exceeds_line_cap_{MAX_LINE_ITEM_AMOUNT}"

    # Check if source text is a suspect pattern
    if source_text and is_suspect_numeric(source_text):
        suspect_type = classify_suspect_numeric(source_text)
        return False, f"suspect_pattern_{suspect_type}"

    # Require row context for amounts
    if not row_has_description:
        return False, "no_row_context"

    return True, None


def validate_qty_rate_amount(
    qty: Optional[float],
    rate: Optional[float],
    amount: Optional[float],
    tolerance: float = 0.01,
) -> Tuple[bool, Optional[str]]:
    """Validate that amount aligns with qty × rate within tolerance.

    Args:
        qty: Quantity
        rate: Unit rate
        amount: Total amount
        tolerance: Acceptable relative difference (default 1%)

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    if qty is None or rate is None or amount is None:
        # Cannot validate without all values; allow by default
        return True, None

    if qty <= 0 or rate <= 0:
        return True, None  # Cannot validate

    expected = qty * rate

    if expected == 0:
        return True, None

    relative_diff = abs(amount - expected) / expected

    if relative_diff > tolerance:
        return False, f"qty_rate_mismatch_expected_{expected:.2f}_got_{amount:.2f}"

    return True, None


def validate_grand_total(total: float) -> Tuple[bool, Optional[str]]:
    """Validate the grand total against sanity caps.

    Args:
        total: The computed grand total

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    if total < 0:
        return False, "negative_total"

    if total > MAX_GRAND_TOTAL:
        return False, f"exceeds_grand_cap_{MAX_GRAND_TOTAL}"

    return True, None


# =============================================================================
# Row Context Validation
# =============================================================================
def has_valid_row_context(
    description: str,
    columns: list,
    min_description_len: int = 3,
    min_columns: int = 2,
) -> bool:
    """Check if a row has sufficient context to be a valid item.

    Args:
        description: Item description text
        columns: List of column values in the row
        min_description_len: Minimum description length
        min_columns: Minimum number of columns (desc + amount)

    Returns:
        True if row has valid context for item extraction
    """
    if not description or len(description.strip()) < min_description_len:
        return False

    if len(columns) < min_columns:
        return False

    # Description should have at least some alphabetic characters
    if not re.search(r"[a-zA-Z]{2,}", description):
        return False

    return True
