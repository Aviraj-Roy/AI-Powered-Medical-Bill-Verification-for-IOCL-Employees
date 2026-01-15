
MEDICINE_KEYWORDS = ["tablet", "capsule", "syrup", "mg"]
TEST_KEYWORDS = ["x-ray", "scan", "mri", "blood"]
PROCEDURE_KEYWORDS = ["surgery", "procedure", "operation"]
SERVICE_KEYWORDS = ["consultation", "room", "service", "ward"]


def classify_items(items):
    classified = {
        "medicines": [],
        "tests": [],
        "procedures": [],
        "services": []
    }

    for item in items:
        desc = item["description"].lower()

        if any(k in desc for k in MEDICINE_KEYWORDS):
            classified["medicines"].append(item)
        elif any(k in desc for k in TEST_KEYWORDS):
            classified["tests"].append(item)
        elif any(k in desc for k in PROCEDURE_KEYWORDS):
            classified["procedures"].append(item)
        else:
            # Everything else is treated as a service
            classified["services"].append(item)

    return classified
