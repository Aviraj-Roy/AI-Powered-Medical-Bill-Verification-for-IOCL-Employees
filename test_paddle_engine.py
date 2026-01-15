import os
from app.ocr.paddle_engine import run_ocr


def test_ocr_processing():
    print(" Starting OCR Test ---\n")

    images_dir = "uploads/processed"

    # Check if directory exists
    if not os.path.exists(images_dir):
        print(f" ERROR: Directory not found: {images_dir}")
        print(f"   Current working directory: {os.getcwd()}")
        return

    # Find all bill page images
    all_files = os.listdir(images_dir)
    image_paths = sorted([
        os.path.join(images_dir, f)
        for f in all_files
        if f.startswith("Bill_page_") and f.endswith(".png")
    ])

    if not image_paths:
        print(f" No bill page images found in {images_dir}")
        print(f"   Files in directory: {all_files}")
        return

    print(f"📄 Found {len(image_paths)} pages:")
    for path in image_paths:
        file_size = os.path.getsize(path)
        print(f"   - {path} ({file_size:,} bytes)")
    print()

    try:
        result = run_ocr(image_paths)

        if not result["raw_text"].strip():
            print("  WARNING: No text was extracted!")
            print("   This could mean:")
            print("   - Images are corrupted or empty")
            print("   - Images are not readable (wrong format)")
            print("   - PaddleOCR installation issue")
            return

        print("\n" + "="*60)
        print("RAW OCR TEXT")
        print("="*60 + "\n")
        print(result["raw_text"])

        print(f"\n" + "="*60)
        print(f"GROUPED BILL ITEMS ({len(result['item_blocks'])} blocks)")
        print("="*60 + "\n")

        for idx, block in enumerate(result["item_blocks"], start=1):
            text = block["text"].strip()
            if not text:
                continue

            confidence_avg = sum(line["confidence"] for line in block["lines"]) / len(block["lines"])
            
            print(f"[BLOCK {idx}] (confidence: {confidence_avg:.2f})")
            print(text)
            print("-" * 60)

        # Save results to file
        output_file = "ocr_results.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["raw_text"])
        print(f"\n Results saved to: {output_file}")

    except Exception as e:
        print(f" ERROR during OCR processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ocr_processing()