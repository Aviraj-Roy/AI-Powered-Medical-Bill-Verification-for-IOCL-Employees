"""Verification tests for refactored bill extraction pipeline.

Tests the key fixes:
1. Header → Item Leakage prevention
2. Section context tracking
3. Multi-page header locking
4. Numeric guardrails
5. Payment isolation
"""

import sys
sys.path.insert(0, ".")

from app.extraction.bill_extractor import (
    BillExtractor,
    extract_bill_data,
    is_paymentish,
    extract_amount_from_text,
)
from app.extraction.numeric_guards import (
    is_suspect_numeric,
    validate_amount,
    classify_suspect_numeric,
)
from app.extraction.zone_detector import (
    is_header_label,
    is_payment_zone,
    should_skip_as_header_label,
)
from app.extraction.section_tracker import (
    SectionTracker,
    detect_section_header,
    classify_item_by_description,
)


def test_numeric_guardrails():
    """Test that phone numbers, MRNs, dates are rejected."""
    print("Testing numeric guardrails...")

    # Phone numbers should be suspect
    assert is_suspect_numeric("9876543210"), "Phone number not detected"
    assert is_suspect_numeric("+919876543210"), "Phone with country code not detected"

    # MRNs should be suspect
    assert is_suspect_numeric("10010001143682"), "12-digit MRN not detected"
    assert is_suspect_numeric("MRN1234567890"), "MRN with prefix not detected"

    # Dates should be suspect
    assert is_suspect_numeric("17/01/2026"), "Date not detected"
    assert is_suspect_numeric("2026-01-17"), "ISO date not detected"

    # Valid amounts should NOT be suspect
    assert not is_suspect_numeric("1234.56"), "Valid amount incorrectly flagged"
    assert not is_suspect_numeric("10,000.00"), "Valid amount incorrectly flagged"

    # Amount extraction should reject suspect values
    assert extract_amount_from_text("9876543210") is None, "Phone parsed as amount"
    assert extract_amount_from_text("10010001143682") is None, "MRN parsed as amount"

    print("  ✓ Numeric guardrails working")


def test_header_label_detection():
    """Test that header labels are correctly identified."""
    print("Testing header label detection...")

    # These should be detected as header labels
    assert is_header_label("Patient Name:"), "Patient Name not detected"
    assert is_header_label("Patient MRN"), "Patient MRN not detected"
    assert is_header_label("Gender/Age"), "Gender/Age not detected"
    assert is_header_label("Address:"), "Address not detected"
    assert is_header_label("Bill No:"), "Bill No not detected"
    assert is_header_label("UHID:"), "UHID not detected"

    # These should NOT be header labels
    assert not is_header_label("TILT TABLE TEST"), "Medical test incorrectly flagged"
    assert not is_header_label("MRI BRAIN"), "MRI incorrectly flagged"
    assert not is_header_label("CONSULTATION"), "Consultation incorrectly flagged"

    print("  ✓ Header label detection working")


def test_payment_detection():
    """Test that payment indicators are correctly detected."""
    print("Testing payment detection...")

    # These should be detected as payments
    assert is_paymentish("RCPO-12345"), "RCPO not detected"
    assert is_paymentish("Receipt No: 12345"), "Receipt not detected"
    assert is_paymentish("CASH PAYMENT"), "Cash payment not detected"
    assert is_paymentish("UTR: 123456789012"), "UTR not detected"
    assert is_paymentish("Total Paid"), "Total Paid not detected"

    # These should NOT be payments
    assert not is_paymentish("TILT TABLE TEST"), "Medical test incorrectly flagged"
    assert not is_paymentish("MRI BRAIN"), "MRI incorrectly flagged"
    assert not is_paymentish("BLOOD TEST"), "Blood test incorrectly flagged"

    print("  ✓ Payment detection working")


def test_section_detection():
    """Test section header detection."""
    print("Testing section detection...")

    # These should be detected as section headers
    assert detect_section_header("DIAGNOSTICS") == "diagnostics_tests"
    assert detect_section_header("--- RADIOLOGY ---") == "radiology"
    assert detect_section_header("CONSULTATION") == "consultation"
    assert detect_section_header("Medicines") == "medicines"
    assert detect_section_header("Laboratory Services") == "diagnostics_tests"

    # These should NOT be section headers (too long or have amounts)
    assert detect_section_header("TILT TABLE TEST ₹5,000.00") is None

    print("  ✓ Section detection working")


def test_item_classification():
    """Test item classification by description."""
    print("Testing item classification...")

    # Test classification by description
    assert classify_item_by_description("TILT TABLE TEST") == "diagnostics_tests"
    assert classify_item_by_description("MRI BRAIN") == "radiology"
    assert classify_item_by_description("CT SCAN CHEST") == "radiology"
    assert classify_item_by_description("CONSULTATION DR SMITH") == "consultation"
    assert classify_item_by_description("TABLET PARACETAMOL 500MG") == "medicines"
    assert classify_item_by_description("ROOM CHARGES - DELUXE") == "hospitalization"

    print("  ✓ Item classification working")


def test_section_tracker_persistence():
    """Test that section context persists across pages."""
    print("Testing section tracker persistence...")

    tracker = SectionTracker()

    # Add section on page 0
    tracker.add_event(page=0, y=100.0, section="diagnostics_tests", text="DIAGNOSTICS")

    # Item on page 0 after section should get that section
    assert tracker.get_section_at(page=0, y=150.0) == "diagnostics_tests"

    # Item on page 1 (no new section) should STILL get that section
    assert tracker.get_section_at(page=1, y=50.0) == "diagnostics_tests"

    # Add new section on page 1
    tracker.add_event(page=1, y=200.0, section="radiology", text="RADIOLOGY")

    # Item on page 1 after new section should get new section
    assert tracker.get_section_at(page=1, y=250.0) == "radiology"

    # Item on page 2 should get radiology (persists)
    assert tracker.get_section_at(page=2, y=50.0) == "radiology"

    print("  ✓ Section tracker persistence working")


def test_extraction_pipeline():
    """Test the full extraction pipeline with mock OCR data."""
    print("Testing extraction pipeline...")

    # Mock OCR result
    ocr_result = {
        "raw_text": "",
        "lines": [
            # Header zone
            {"text": "Patient Name: John Doe", "page": 0, "box": [[0, 10], [100, 10], [100, 20], [0, 20]], "confidence": 0.9},
            {"text": "Patient MRN: 12345678", "page": 0, "box": [[0, 30], [100, 30], [100, 40], [0, 40]], "confidence": 0.9},
            {"text": "Bill No: BL123456", "page": 0, "box": [[0, 50], [100, 50], [100, 60], [0, 60]], "confidence": 0.9},
            # Table start
            {"text": "S.No", "page": 0, "box": [[0, 100], [50, 100], [50, 110], [0, 110]], "confidence": 0.9},
            {"text": "Description", "page": 0, "box": [[50, 100], [150, 100], [150, 110], [50, 110]], "confidence": 0.9},
            # Section header
            {"text": "DIAGNOSTICS", "page": 0, "box": [[0, 150], [100, 150], [100, 160], [0, 160]], "confidence": 0.9},
            # Item (no amount pattern, so won't be extracted without item_blocks)
        ],
        "item_blocks": [
            {
                "text": "1 TILT TABLE TEST 1 5000.00 5000.00",
                "description": "TILT TABLE TEST",
                "columns": ["1", "5000.00", "5000.00"],
                "page": 0,
                "y": 200.0,
            },
            {
                "text": "2 CONSULTATION DR SMITH 1 1000.00 1000.00",
                "description": "CONSULTATION DR SMITH",
                "columns": ["1", "1000.00", "1000.00"],
                "page": 0,
                "y": 250.0,
            },
        ],
    }

    result = extract_bill_data(ocr_result)

    # Check header extraction
    assert result["patient"]["name"] == "John Doe", f"Patient name wrong: {result['patient']['name']}"
    assert result["header"]["primary_bill_number"] == "BL123456", f"Bill number wrong: {result['header']}"

    # Check that items were extracted (not as headers)
    all_items = []
    for cat, items in result["items"].items():
        all_items.extend(items)

    assert len(all_items) > 0, "No items extracted"

    # Check categorization
    diagnostics_items = result["items"].get("diagnostics_tests", [])
    consultation_items = result["items"].get("consultation", [])

    # TILT TABLE TEST should be in diagnostics (based on description classification)
    tilt_test = [i for i in all_items if "TILT" in i.get("description", "").upper()]
    assert len(tilt_test) > 0, "TILT TABLE TEST not extracted"

    # No payments should leak into items
    assert len(result["payments"]) == 0, "Unexpected payments extracted"

    print("  ✓ Extraction pipeline working")


def test_header_not_in_items():
    """Test that header labels don't appear as items."""
    print("Testing header leakage prevention...")

    ocr_result = {
        "raw_text": "",
        "lines": [
            {"text": "Patient Name: John Doe", "page": 0, "box": [[0, 10], [100, 10], [100, 20], [0, 20]], "confidence": 0.9},
            {"text": "Gender|Age|DoB", "page": 0, "box": [[0, 30], [100, 30], [100, 40], [0, 40]], "confidence": 0.9},
            {"text": "Address: 123 Main St", "page": 0, "box": [[0, 50], [100, 50], [100, 60], [0, 60]], "confidence": 0.9},
        ],
        "item_blocks": [],
    }

    result = extract_bill_data(ocr_result)

    # Check that header labels are NOT in items
    all_items = []
    for cat, items in result["items"].items():
        all_items.extend(items)

    for item in all_items:
        desc = item.get("description", "").lower()
        assert "patient name" not in desc, f"'Patient Name' leaked into items: {desc}"
        assert "gender|age" not in desc, f"'Gender|Age' leaked into items: {desc}"
        assert "address" not in desc, f"'Address' leaked into items: {desc}"

    print("  ✓ Header leakage prevention working")


def test_payment_isolation():
    """Test that payments don't leak into medical items."""
    print("Testing payment isolation...")

    ocr_result = {
        "raw_text": "",
        "lines": [
            {"text": "S.No", "page": 0, "box": [[0, 100], [50, 100], [50, 110], [0, 110]], "confidence": 0.9},
        ],
        "item_blocks": [
            {
                "text": "RCPO-12345 CASH ₹5,000.00",
                "description": "RCPO-12345 CASH",
                "columns": ["5,000.00"],
                "page": 0,
                "y": 200.0,
            },
            {
                "text": "Total Paid ₹5,000.00",
                "description": "Total Paid",
                "columns": ["5,000.00"],
                "page": 0,
                "y": 250.0,
            },
        ],
    }

    result = extract_bill_data(ocr_result)

    # Check that payments are in payments[], not items
    all_items = []
    for cat, items in result["items"].items():
        all_items.extend(items)

    for item in all_items:
        desc = item.get("description", "").upper()
        assert "RCPO" not in desc, f"Payment RCPO leaked into items: {desc}"
        assert "TOTAL PAID" not in desc, f"Total Paid leaked into items: {desc}"

    # Payments should be captured
    assert len(result["payments"]) >= 1, "Payments not captured"

    print("  ✓ Payment isolation working")


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("Running Refactored Bill Extractor Verification Tests")
    print("=" * 60 + "\n")

    tests = [
        test_numeric_guardrails,
        test_header_label_detection,
        test_payment_detection,
        test_section_detection,
        test_item_classification,
        test_section_tracker_persistence,
        test_extraction_pipeline,
        test_header_not_in_items,
        test_payment_isolation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
