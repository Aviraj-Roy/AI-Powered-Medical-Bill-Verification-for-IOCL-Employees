from paddleocr import PaddleOCR
import os
import traceback
import numpy as np

# -------------------------
# OCR INIT (PaddleOCR 3.3.2)
# -------------------------
ocr = PaddleOCR(use_angle_cls=True, lang="en")


# -------------------------
# Geometry helpers
# -------------------------
def _top_y(box):
    if box is None:
        return 0.0
    try:
        if isinstance(box, np.ndarray):
            if box.size == 0:
                return 0.0
            return float(box[:, 1].min())
        if isinstance(box, (list, tuple)) and len(box) > 0:
            return float(min(p[1] for p in box))
    except Exception:
        pass
    return 0.0


def _left_x(box):
    if box is None:
        return 0.0
    try:
        if isinstance(box, np.ndarray):
            if box.size == 0:
                return 0.0
            return float(box[:, 0].min())
        if isinstance(box, (list, tuple)) and len(box) > 0:
            return float(min(p[0] for p in box))
    except Exception:
        pass
    return 0.0


def _height(box):
    if box is None:
        return 0.0
    try:
        if isinstance(box, np.ndarray):
            if box.size == 0:
                return 0.0
            ys = box[:, 1]
            return float(ys.max() - ys.min())
        if isinstance(box, (list, tuple)) and len(box) > 0:
            ys = [p[1] for p in box]
            return float(max(ys) - min(ys))
    except Exception:
        pass
    return 0.0


# -------------------------
# Normalize PaddleOCR output
# -------------------------
def _normalize_page(page_res):
    lines = []

    if isinstance(page_res, dict) and "rec_texts" in page_res:
        texts = page_res.get("rec_texts", [])
        scores = page_res.get("rec_scores", [])
        boxes = page_res.get("rec_polys", [])

        for i, text in enumerate(texts):
            if not text.strip():
                continue
            lines.append({
                "text": text.strip(),
                "confidence": float(scores[i]) if i < len(scores) else 1.0,
                "box": boxes[i]
            })

    return lines


# -------------------------
# Row clustering (Y-axis)
# -------------------------
def _cluster_rows(lines):
    if not lines:
        return []

    for l in lines:
        l["_y"] = _top_y(l["box"])
        l["_h"] = _height(l["box"])

    avg_height = sum(l["_h"] for l in lines) / max(len(lines), 1)
    row_threshold = avg_height * 0.8 if avg_height > 0 else 15

    lines.sort(key=lambda x: x["_y"])

    rows = []
    current = [lines[0]]

    for line in lines[1:]:
        prev = current[-1]
        if abs(line["_y"] - prev["_y"]) <= row_threshold:
            current.append(line)
        else:
            rows.append(current)
            current = [line]

    rows.append(current)

    for l in lines:
        l.pop("_y", None)
        l.pop("_h", None)

    return rows


# -------------------------
# Column segmentation
# -------------------------
def _split_columns(row, date_x):
    row.sort(key=lambda l: _left_x(l["box"]))

    description_parts = []
    numeric_parts = []

    for line in row:
        if _left_x(line["box"]) < date_x:
            description_parts.append(line["text"])
        else:
            numeric_parts.append(line["text"])

    return description_parts, numeric_parts


# -------------------------
# Main OCR pipeline
# -------------------------
def run_ocr(img_paths):
    """
    Multi-page OCR with semantic bill-item grouping
    """
    if isinstance(img_paths, str):
        img_paths = [img_paths]

    all_lines = []
    item_blocks = []

    # -------- OCR --------
    for img_path in img_paths:
        try:
            results = ocr.predict(os.path.abspath(img_path))

            if hasattr(results, "to_dict"):
                results = results.to_dict()

            for page in results:
                page_lines = _normalize_page(page)
                all_lines.extend(page_lines)

        except Exception:
            traceback.print_exc()

    if not all_lines:
        return {"raw_text": "", "lines": [], "item_blocks": []}

    raw_text = "\n".join(l["text"] for l in all_lines)

    # -------- Grouping --------
    rows = _cluster_rows(all_lines)

    # Estimate DATE column X (anchor)
    date_candidates = [
        _left_x(l["box"])
        for l in all_lines
        if "-" in l["text"] and len(l["text"]) == 10
    ]
    date_x = min(date_candidates) if date_candidates else 250

    for row in rows:
        desc, nums = _split_columns(row, date_x)

        if not desc or not nums:
            continue

        item_blocks.append({
            "text": " ".join(desc + nums),
            "description": " ".join(desc),
            "columns": nums,
            "lines": row
        })

    return {
        "raw_text": raw_text,
        "lines": all_lines,
        "item_blocks": item_blocks
    }
