import os
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Dict, List, Optional
from datetime import datetime

load_dotenv()


class MongoDBClient:
    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME", "medical_bills")
        self.collection_name = os.getenv("MONGO_COLLECTION_NAME", "bills")

        if not self.mongo_uri:
            raise ValueError("MONGO_URI not found in .env")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        # Create indexes for better query performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes on commonly queried fields."""
        self.collection.create_index("bill_headers.patient_mrn")
        self.collection.create_index("bill_headers.bill_number")
        self.collection.create_index("extraction_date")
    
    def insert_bill(self, bill_data: Dict) -> str:
        """
        Insert a medical bill document into MongoDB.
        
        Args:
            bill_data: Structured bill data from extractor
        
        Returns:
            str: Inserted document ID
        """
        # Add metadata
        bill_data["inserted_at"] = datetime.now().isoformat()
        
        result = self.collection.insert_one(bill_data)
        return str(result.inserted_id)
    
    def get_bill_by_id(self, bill_id: str) -> Optional[Dict]:
        """
        Fetch a bill by MongoDB ObjectId.
        
        Args:
            bill_id: MongoDB ObjectId as string
        
        Returns:
            Dict: Bill document or None
        """
        from bson import ObjectId
        return self.collection.find_one({"_id": ObjectId(bill_id)})
    
    def get_bills_by_patient_mrn(self, mrn: str) -> List[Dict]:
        """
        Fetch all bills for a patient by MRN.
        
        Args:
            mrn: Patient MRN number
        
        Returns:
            List[Dict]: List of bill documents
        """
        return list(self.collection.find({"bill_headers.patient_mrn": mrn}))
    
    def get_bills_by_patient_name(self, patient_name: str) -> List[Dict]:
        """
        Fetch all bills for a patient by name (case-insensitive).
        
        Args:
            patient_name: Patient name
        
        Returns:
            List[Dict]: List of bill documents
        """
        return list(self.collection.find({
            "bill_headers.patient_name": {"$regex": patient_name, "$options": "i"}
        }))
    
    def get_medicine_summary(self, bill_id: str) -> Dict:
        """
        Get summary of medicines from a bill.
        
        Args:
            bill_id: MongoDB ObjectId as string
        
        Returns:
            Dict: Medicine items and subtotal
        """
        from bson import ObjectId
        bill = self.collection.find_one(
            {"_id": ObjectId(bill_id)},
            {"items.medicines": 1, "subtotals.medicines": 1}
        )
        
        if not bill:
            return None
        
        return {
            "medicines": bill.get("items", {}).get("medicines", []),
            "total": bill.get("subtotals", {}).get("medicines", 0)
        }
    
    def get_category_summary(self, bill_id: str, category: str) -> Dict:
        """
        Get summary of any category from a bill.
        
        Args:
            bill_id: MongoDB ObjectId as string
            category: Category name (e.g., 'medicines', 'consultation', 'packages', 'diagnostics_tests')
        
        Returns:
            Dict: Category items and subtotal
        """
        from bson import ObjectId
        
        projection = {
            f"items.{category}": 1,
            f"subtotals.{category}": 1
        }
        
        bill = self.collection.find_one({"_id": ObjectId(bill_id)}, projection)
        
        if not bill:
            return None
        
        return {
            "items": bill.get("items", {}).get(category, []),
            "total": bill.get("subtotals", {}).get(category, 0)
        }
    
    def get_patient_total_spending(self, mrn: str) -> Dict:
        """
        Calculate total spending across all bills for a patient.
        
        Args:
            mrn: Patient MRN number
        
        Returns:
            Dict: Breakdown by category and grand total
        """
        pipeline = [
            {"$match": {"bill_headers.patient_mrn": mrn}},
            {"$group": {
                "_id": None,
                "total_medicines": {"$sum": "$subtotals.medicines"},
                "total_consultation": {"$sum": "$subtotals.consultation"},
                "total_packages": {"$sum": "$subtotals.packages"},
                "total_administrative": {"$sum": "$subtotals.administrative"},
                "total_implants_devices": {"$sum": "$subtotals.implants_devices"},
                "total_surgical_consumables": {"$sum": "$subtotals.surgical_consumables"},
                "total_hospitalization": {"$sum": "$subtotals.hospitalization"},
                "total_diagnostics_tests": {"$sum": "$subtotals.diagnostics_tests"},
                "grand_total": {"$sum": "$grand_total"}
            }}
        ]
        
        result = list(self.collection.aggregate(pipeline))
        
        if not result:
            return {"message": "No bills found for this patient"}
        
        return result[0]
    
    def search_medicine_across_bills(self, medicine_keyword: str) -> List[Dict]:
        """
        Search for a specific medicine across all bills.
        
        Args:
            medicine_keyword: Medicine name or keyword
        
        Returns:
            List[Dict]: Bills containing the medicine
        """
        return list(self.collection.find({
            "items.medicines.description": {"$regex": medicine_keyword, "$options": "i"}
        }))
    
    def search_diagnostic_test_across_bills(self, test_keyword: str) -> List[Dict]:
        """
        Search for a specific diagnostic test across all bills.
        
        Args:
            test_keyword: Test name or keyword (e.g., 'MRI', 'blood test', 'X-ray')
        
        Returns:
            List[Dict]: Bills containing the diagnostic test
        """
        return list(self.collection.find({
            "items.diagnostics_tests.description": {"$regex": test_keyword, "$options": "i"}
        }))
    
    def get_statistics(self) -> Dict:
        """
        Get overall statistics from the collection.
        
        Returns:
            Dict: Statistics summary
        """
        pipeline = [
            {"$group": {
                "_id": None,
                "total_bills": {"$sum": 1},
                "total_revenue": {"$sum": "$grand_total"},
                "avg_bill_amount": {"$avg": "$grand_total"}
            }}
        ]
        
        result = list(self.collection.aggregate(pipeline))
        
        if not result:
            return {"message": "No data available"}
        
        return result[0]