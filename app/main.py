from app.ingestion.pdf_loader import pdf_to_images
from app.ocr.image_preprocessor import preprocess_image
from app.ocr.paddle_engine import run_ocr
from app.extraction.bill_extractor import extract_bill_data
from app.db.mongo_client import MongoDBClient


def process_bill(pdf_path: str):
    # 1. Convert PDF → Images
    image_paths = pdf_to_images(pdf_path)

    db = MongoDBClient()

    for image_path in image_paths:
        # 2. Preprocess Image
        processed_image = preprocess_image(image_path)

        # 3. OCR
        ocr_text = run_ocr(processed_image)

        # 4. Extract Bill Fields
        bill_data = extract_bill_data(ocr_text)

        # 5. Store in MongoDB
        bill_id = db.insert_bill(bill_data)
        print(f"Stored bill with ID: {bill_id}")


if __name__ == "__main__":
    process_bill("M_Bill.pdf")
