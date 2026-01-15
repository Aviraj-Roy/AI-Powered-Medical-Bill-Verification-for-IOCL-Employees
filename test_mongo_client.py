from app.db.mongo_client import MongoDBClient

client = MongoDBClient()

sample_bill = {
    "patient_name": "John Doe",
    "hospital_name": "City Medical Centre",
    "total_amount": 895.00,
    "line_items": {
        "medicines": [{"description": "Paracetamol", "amount": 50.0}],
        "tests": [],
        "procedures": [],
        "services": [{"description": "Consultation", "amount": 500.0}]
    }
}

bill_id = client.insert_bill(sample_bill)
print("Inserted Bill ID:", bill_id)

retrieved = client.get_bill_by_id(bill_id)
print("Retrieved Bill:", retrieved)
