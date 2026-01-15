import re
from typing import Dict, List, Optional
from datetime import datetime


class BillExtractor:
    """
    Extracts structured data from medical bills with flexible section detection.
    """
    
    # Section keyword mappings (flexible matching)
    SECTION_KEYWORDS = {
        "administrative": ["administrative", "admission", "processing", "record"],
        "packages": ["package", "pkg"],
        "consultation": ["consultation", "consult"],
        "implants_devices": ["implant", "device", "stent", "coronary"],
        "medicines": ["medicine", "tablet", "capsule", "injection", "syrup", "regulated pricing"],
        "surgical_consumables": ["surgical consumable", "consumable", "gauze", "syringe", "gloves"],
        "hospitalization": ["hospitalisation", "hospitalization", "room", "ward", "bed", "care"],
        "diagnostics_tests": ["diagnostic", "diagnostics", "test", "tests", "lab", "laboratory", 
                              "pathology", "radiology", "x-ray", "xray", "scan", "mri", "ct scan", 
                              "ultrasound", "sonography", "ecg", "ekg", "echo", "blood test", 
                              "urine test", "biopsy", "culture", "screening", "imaging", "investigation"]
    }
    
    def __init__(self):
        self.current_section = None
    
    def extract_bill_header(self, text: str) -> Dict:
        """Extract basic bill information."""
        header = {}
        
        # Bill Number
        bill_match = re.search(r"Bill\s*No\s*[:.]?\s*([A-Z0-9]+)", text, re.IGNORECASE)
        if bill_match:
            header["bill_number"] = bill_match.group(1).strip()
        
        # Patient Name
        name_match = re.search(r"Patient\s*Name\s*[:.]?\s*([^\n]+)", text, re.IGNORECASE)
        if name_match:
            header["patient_name"] = name_match.group(1).strip()
        
        # Patient MRN
        mrn_match = re.search(r"Patient\s*MRN\s*[:.]?\s*([0-9]+)", text, re.IGNORECASE)
        if mrn_match:
            header["patient_mrn"] = mrn_match.group(1).strip()
        
        # Hospital/Centre Name
        hospital_match = re.search(r"^([^\n]+(?:Hospital|Centre|Clinic|Medical)[^\n]*)", text, re.MULTILINE | re.IGNORECASE)
        if hospital_match:
            header["hospital_name"] = hospital_match.group(1).strip()
        
        # Billing Date
        date_match = re.search(r"Billing\s*Date\s*[:.]?\s*([0-9\-/]+)", text, re.IGNORECASE)
        if date_match:
            header["billing_date"] = date_match.group(1).strip()
        
        # Total Amount (Patient Payable)
        total_match = re.search(r"Patient\s*Payable\s*[:.]?\s*₹?\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
        if total_match:
            amount_str = total_match.group(1).replace(",", "")
            header["total_amount"] = float(amount_str)
        
        # Discount
        discount_match = re.search(r"Less\s*Discount\s*[:.]?\s*₹?\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
        if discount_match:
            amount_str = discount_match.group(1).replace(",", "")
            header["discount"] = float(amount_str)
        
        # Gender and Age
        gender_age_match = re.search(r"Gender\|Age\|DoB\s*[:.]?\s*([^|]+)\|([^|]+)\|", text, re.IGNORECASE)
        if gender_age_match:
            header["gender"] = gender_age_match.group(1).strip()
            header["age"] = gender_age_match.group(2).strip()
        
        return header
    
    def identify_section(self, line: str) -> Optional[str]:
        """
        Identify which section a line belongs to based on keywords.
        Returns section name or None.
        """
        line_lower = line.lower().strip()
        
        # Check if line is a section header (usually uppercase or bold)
        for section, keywords in self.SECTION_KEYWORDS.items():
            if any(keyword in line_lower for keyword in keywords):
                # Avoid false positives - check if it's likely a header
                if len(line.strip()) < 50 and not re.search(r"\d+\.\d{2}$", line):
                    return section
        
        return None
    
    def extract_line_item(self, line: str) -> Optional[Dict]:
        """
        Extract a line item (description + amount) from a line.
        Returns dict with description and amount, or None if not a valid item.
        """
        # Look for amount at the end of line (format: 1,234.56 or 1234.56)
        amount_match = re.search(r"₹?\s*([\d,]+\.\d{2})\s*$", line)
        
        if not amount_match:
            return None
        
        amount_str = amount_match.group(1).replace(",", "")
        amount = float(amount_str)
        
        # Extract description (everything before the amount)
        description = line[:amount_match.start()].strip()
        
        # Clean up description - remove date patterns, leading numbers/dots
        description = re.sub(r"^\d+\.\s*", "", description)  # Remove "1. "
        description = re.sub(r"\d{2}-\d{2}-\d{4}", "", description)  # Remove dates
        description = re.sub(r"\s+", " ", description).strip()  # Clean whitespace
        
        # Skip if description is too short or looks like a header
        if len(description) < 3 or description.lower() in ["total", "subtotal"]:
            return None
        
        return {
            "description": description,
            "amount": amount
        }
    
    def extract_items_by_section(self, text: str) -> Dict[str, List[Dict]]:
        """
        Parse the entire bill text and group items by section.
        """
        sections = {
            "administrative": [],
            "packages": [],
            "consultation": [],
            "implants_devices": [],
            "medicines": [],
            "surgical_consumables": [],
            "hospitalization": [],
            "diagnostics_tests": []
        }
        
        lines = text.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Check if this line is a section header
            detected_section = self.identify_section(line)
            if detected_section:
                current_section = detected_section
                continue
            
            # Try to extract line item
            if current_section:
                item = self.extract_line_item(line)
                if item:
                    sections[current_section].append(item)
        
        return sections
    
    def calculate_subtotals(self, sections: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate subtotals for each section."""
        subtotals = {}
        
        for section_name, items in sections.items():
            subtotals[section_name] = sum(item["amount"] for item in items)
        
        return subtotals
    
    def extract_all_bills(self, ocr_text: str) -> Dict:
        """
        Main extraction method - processes entire OCR text (may contain multiple bills).
        Combines all bills into one structured document.
        """
        # Split by common bill separators
        bill_texts = re.split(r"(?=BILL AND RECEIPT|FINAL BILL)", ocr_text, flags=re.IGNORECASE)
        
        # Storage for combined data
        all_headers = []
        combined_sections = {
            "administrative": [],
            "packages": [],
            "consultation": [],
            "implants_devices": [],
            "medicines": [],
            "surgical_consumables": [],
            "hospitalization": [],
            "diagnostics_tests": []
        }
        
        # Process each bill
        for bill_text in bill_texts:
            if len(bill_text.strip()) < 100:  # Skip empty/tiny fragments
                continue
            
            # Extract header
            header = self.extract_bill_header(bill_text)
            if header:
                all_headers.append(header)
            
            # Extract items by section
            sections = self.extract_items_by_section(bill_text)
            
            # Combine items
            for section_name, items in sections.items():
                combined_sections[section_name].extend(items)
        
        # Calculate subtotals
        subtotals = self.calculate_subtotals(combined_sections)
        
        # Build final document
        result = {
            "extraction_date": datetime.now().isoformat(),
            "bills_processed": len(all_headers),
            "bill_headers": all_headers,
            "items": combined_sections,
            "subtotals": subtotals,
            "grand_total": sum(subtotals.values())
        }
        
        return result


def extract_bill_data(ocr_result: Dict) -> Dict:
    """
    Main entry point for bill extraction.
    
    Args:
        ocr_result: Dict containing 'raw_text' from OCR
    
    Returns:
        Dict: Structured bill data ready for MongoDB
    """
    extractor = BillExtractor()
    raw_text = ocr_result.get("raw_text", "")
    
    if not raw_text:
        raise ValueError("No raw_text found in ocr_result")
    
    bill_data = extractor.extract_all_bills(raw_text)
    
    return bill_data