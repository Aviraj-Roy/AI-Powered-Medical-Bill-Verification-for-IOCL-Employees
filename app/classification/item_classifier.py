"""
Medical Bill Item Classifier

Classifies line items into categories using keyword matching and patterns.
"""
import re
from typing import Dict, List, Optional


# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================
CATEGORY_RULES = {
    "medicines": {
        "keywords": [
            "tablet", "capsule", "syrup", "injection", "infusion",
            "solution", "ointment", "cream", "gel", "drops",
            "inhaler", "spray", "suspension", "powder",
            # Common drug suffixes/types
            "antibiotic", "analgesic", "antipyretic", "antacid",
        ],
        "patterns": [
            r"\d+\s*mg",      # Dosage: 500mg, 250 mg
            r"\d+\s*ml",      # Volume: 100ml
            r"\d+\s*mcg",     # Micrograms
            r"\d+\s*iu",      # International units
            r"\d+\s*gm?",     # Grams: 5g, 5gm
            r"\d+%",          # Percentage solutions: 5%
        ],
        "priority": 1,
    },
    
    "regulated_pricing_drugs": {
        "keywords": [
            "regulated pricing", "dpco", "nlem",
            "contrast", "iohexol", "omnipaque",  # Contrast media
            "heparin", "insulin",
        ],
        "patterns": [],
        "priority": 0,  # Higher priority - check first
    },
    
    "surgical_consumables": {
        "keywords": [
            "gloves", "syringe", "needle", "catheter", "cannula",
            "bandage", "gauze", "drape", "dressing", "swab",
            "mask", "gown", "cap", "cover", "screen cover",
            "iv set", "iv catheter", "stop cock", "extension",
            "electrode", "ecg electrode", "blade", "surgical blade",
            "urinal", "bed pan", "thermometer", "wipes",
            "introducer", "hand care", "sterile",
        ],
        "patterns": [
            r"\d+g\s*x",      # Needle gauge: 23G x 1.5
            r"size\s*\d",     # Glove sizes
        ],
        "priority": 2,
    },
    
    "implants_devices": {
        "keywords": [
            "stent", "implant", "pacemaker", "defibrillator",
            "guide wire", "guidewire", "guiding catheter",
            "ptca", "balloon", "angioplasty",
            "prosthesis", "mesh", "plate", "screw",
            "coronary", "vascular",
        ],
        "patterns": [
            r"\d+fr",         # French size for catheters
            r"\d+\.\d+\s*x\s*\d+",  # Dimensions: 0.014 x 190cm
        ],
        "priority": 1,
    },
    
    "diagnostics_tests": {
        "keywords": [
            "x-ray", "xray", "scan", "ct scan", "mri", "pet",
            "ultrasound", "usg", "sonography", "echo", "echocardiogram",
            "ecg", "ekg", "electrocardiogram",
            "blood test", "urine test", "stool test",
            "pathology", "laboratory", "lab", "culture",
            "biopsy", "histopathology", "cytology",
            "screening", "investigation", "diagnostic",
            "hemoglobin", "hb", "cbc", "lipid", "thyroid",
            "liver function", "kidney function", "lft", "kft", "rft",
            "hba1c", "glucose", "creatinine", "urea", "test",
        ],
        "patterns": [],
        "priority": 3,
    },
    
    "consultation": {
        "keywords": [
            "consultation", "consult", "visit", "first visit", "revisit",
            "follow up", "follow-up", "opinion", "second opinion",
            "doctor fee", "physician fee", "specialist fee",
        ],
        "patterns": [
            r"dr\.?\s+[a-z]+",  # Doctor name reference
        ],
        "priority": 2,
    },
    
    "hospitalization": {
        "keywords": [
            "room", "ward", "bed", "icu", "nicu", "picu", "ccu",
            "nursing", "care", "accommodation", "stay",
            "general ward", "semi private", "private room", "deluxe",
            "hospitalisation", "hospitalization",
        ],
        "patterns": [
            r"room\s*charge",
            r"bed\s*charge",
        ],
        "priority": 3,
    },
    
    "packages": {
        "keywords": [
            "package", "pkg", "bundle", "combo",
            "health checkup", "master health", "executive checkup",
            "angiography package", "angioplasty package",
            "surgery package", "delivery package",
        ],
        "patterns": [],
        "priority": 0,  # High priority
    },
    
    "procedures": {
        "keywords": [
            "surgery", "operation", "procedure",
            "angiography", "angioplasty", "bypass", "cabg",
            "endoscopy", "colonoscopy", "laparoscopy",
            "dialysis", "chemotherapy", "radiotherapy",
            "biopsy procedure", "excision", "incision",
            "catheterization", "cath lab",
        ],
        "patterns": [],
        "priority": 2,
    },
    
    "administrative": {
        "keywords": [
            "administrative", "admin", "registration", "admission",
            "processing", "documentation", "record", "file",
            "discharge", "certificate",
        ],
        "patterns": [],
        "priority": 4,
    },
}


# =============================================================================
# CLASSIFIER CLASS
# =============================================================================
class ItemClassifier:
    """
    Classifies medical bill line items into categories.
    Uses keyword matching and regex patterns with priority ordering.
    """
    
    def __init__(self):
        # Sort categories by priority (lower number = higher priority)
        self.categories = sorted(
            CATEGORY_RULES.items(),
            key=lambda x: x[1].get("priority", 99)
        )
    
    def classify(self, description: str) -> str:
        """
        Classify a single item description.
        
        Args:
            description: Item description text
            
        Returns:
            Category name (str)
        """
        desc_lower = description.lower().strip()
        
        for category, rules in self.categories:
            # Check keywords
            keywords = rules.get("keywords", [])
            if any(kw in desc_lower for kw in keywords):
                return category
            
            # Check patterns
            patterns = rules.get("patterns", [])
            if any(re.search(p, desc_lower) for p in patterns):
                return category
        
        return "other"
    
    def classify_batch(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Classify multiple items and group by category.
        
        Args:
            items: List of item dicts with 'description' key
            
        Returns:
            Dict mapping category names to lists of items
        """
        classified = {cat: [] for cat in CATEGORY_RULES.keys()}
        classified["other"] = []
        
        for item in items:
            desc = item.get("description", "")
            category = self.classify(desc)
            item["category"] = category
            classified[category].append(item)
        
        return classified
    
    def reclassify_with_context(self, items: List[Dict], section_hint: Optional[str] = None) -> List[Dict]:
        """
        Reclassify items using both description and section context.
        
        Args:
            items: List of item dicts
            section_hint: Optional section name from bill structure
            
        Returns:
            Items with updated 'category' field
        """
        for item in items:
            desc = item.get("description", "")
            current_cat = item.get("category", "other")
            
            # If already classified confidently, skip
            if current_cat != "other":
                continue
            
            # Use section hint if available
            if section_hint and section_hint in CATEGORY_RULES:
                item["category"] = section_hint
            else:
                # Try classification
                item["category"] = self.classify(desc)
        
        return items


# =============================================================================
# LEGACY FUNCTION (for backward compatibility)
# =============================================================================
def classify_items(items: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Legacy function for backward compatibility.
    
    Args:
        items: List of item dicts with 'description' key
        
    Returns:
        Dict mapping category names to lists of items
    """
    classifier = ItemClassifier()
    return classifier.classify_batch(items)


def classify_single(description: str) -> str:
    """
    Classify a single item description.
    
    Args:
        description: Item description text
        
    Returns:
        Category name
    """
    classifier = ItemClassifier()
    return classifier.classify(description)
