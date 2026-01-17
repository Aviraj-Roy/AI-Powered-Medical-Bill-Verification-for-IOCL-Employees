from __future__ import annotations

import os
import re
import traceback
from typing import Dict, List, Union

import numpy as np
from paddleocr import PaddleOCR

# -------------------------
# OCR INIT (PaddleOCR 3.3.2)
# -------------------------
ocr = PaddleOCR(use_angle_cls=True, lang="en")


# -------------------------
# Geometry helpers
# -------------------------
def _top_y(box) -> float:
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


def _left_x(box) -> float:
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


def _height(box) -> float:
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
# Normalize PaddleOCR output (single page result)
# -------------------------
def _normalize_page(page_res: Dict, page_number: int) -> List[Dict]:
    lines: List[Dict] = []

    if isinstance(page_res, dict) and "rec_texts" in page_res:
        texts = page_res.get("rec_texts", [])
        scores = page_res.get("rec_scores", [])
        boxes = page_res.get("rec_polys", [])

        for i, text in enumerate(texts):
            if not (text or "").strip():
                continue
            lines.append(
                {
                    "text": text.strip(),
                    "confidence": float(scores[i]) if i < len(scores) else 1.0,
                    "box": boxes[i] if i < len(boxes) else None,
                    "page": page_number,
                }
            )

    return lines


# -------------------------
# Row clustering (Y-axis) - PAGE-AWARE
# -------------------------
def _cluster_rows(lines: List[Dict]) -> List[List[Dict]]:
    """Cluster OCR lines into rows. PAGE-AWARE: rows never span pages."""

    if not lines:
        return []

    # Sort by page then y
    def y_of(l: Dict) -> float:
        return _top_y(l.get("box"))

    lines_sorted = sorted(lines, key=lambda l: (int(l.get("page", 0) or 0), y_of(l)))

    rows: List[List[Dict]] = []
    current: List[Dict] = []
    current_page: int = int(lines_sorted[0].get("page", 0) or 0)

    # compute row thresholds per page (based on average height)
    heights_by_page: Dict[int, List[float]] = {}
    for l in lines_sorted:
        p = int(l.get("page", 0) or 0)
        heights_by_page.setdefault(p, []).append(_height(l.get("box")))

    thresholds: Dict[int, float] = {}
    for p, hs in heights_by_page.items():
        avg_h = (sum(hs) / max(len(hs), 1)) if hs else 0.0
        thresholds[p] = avg_h * 0.8 if avg_h > 0 else 15.0

    for l in lines_sorted:
        p = int(l.get("page", 0) or 0)
        y = y_of(l)

        if not current:
            current = [l]
            current_page = p
            continue

        prev = current[-1]
        prev_y = y_of(prev)

        same_page = p == current_page
        same_row = abs(y - prev_y) <= thresholds.get(p, 15.0)

        if same_page and same_row:
            current.append(l)
        else:
            rows.append(current)
            current = [l]
            current_page = p

    if current:
        rows.append(current)

    return rows


# -------------------------
# Column segmentation
# -------------------------
def _split_columns(row: List[Dict], date_x: float):
    row_sorted = sorted(row, key=lambda l: _left_x(l.get("box")))

    description_parts: List[str] = []
    numeric_parts: List[str] = []

    for line in row_sorted:
        if _left_x(line.get("box")) < date_x:
            description_parts.append(line.get("text", ""))
        else:
            numeric_parts.append(line.get("text", ""))

    return description_parts, numeric_parts


_DATE_LIKE = re.compile(r"\b\d{2}[-/]\d{2}[-/]\d{4}\b")


# -------------------------
# Main OCR pipeline
# -------------------------
def run_ocr(img_paths: Union[str, List[str]]):
    """Multi-page OCR (PaddleOCR) with page-aware line normalization + item grouping.

    Returns:
      {
        raw_text: str,
        lines: [{text, confidence, box, page}],
        item_blocks: [{text, description, columns, lines, page, y}],
        page_count: int,
      }
    """

    if isinstance(img_paths, str):
        img_paths = [img_paths]

    all_lines: List[Dict] = []

    # OCR each page image and tag every line with its page number
    for page_number, img_path in enumerate(img_paths):
        try:
            results = ocr.predict(os.path.abspath(img_path))
            if hasattr(results, "to_dict"):
                results = results.to_dict()

            # PaddleOCR may return a list of page dicts; treat all as belonging to this image.
            for page_res in results:
                all_lines.extend(_normalize_page(page_res, page_number))

        except Exception:
            traceback.print_exc()
            continue

    if not all_lines:
        return {"raw_text": "", "lines": [], "item_blocks": [], "page_count": 0}

    # Build raw_text in reading order with page breaks
    def y_of(l: Dict) -> float:
        return _top_y(l.get("box"))

    all_lines_sorted = sorted(all_lines, key=lambda l: (int(l.get("page", 0) or 0), y_of(l)))

    raw_parts: List[str] = []
    current_page = -1
    for l in all_lines_sorted:
        p = int(l.get("page", 0) or 0)
        if p != current_page:
            if current_page >= 0:
                raw_parts.append(f"\n--- PAGE {p + 1} ---\n")
            current_page = p
        raw_parts.append(l.get("text", ""))

    raw_text = "\n".join(raw_parts)

    # Estimate DATE column X per page (anchor for splitting)
    date_x_by_page: Dict[int, float] = {}
    for l in all_lines_sorted:
        p = int(l.get("page", 0) or 0)
        t = l.get("text", "") or ""
        if _DATE_LIKE.search(t) or ("-" in t and len(t.strip()) == 10):
            x = _left_x(l.get("box"))
            date_x_by_page[p] = min(date_x_by_page.get(p, x), x)

    # Group into item blocks
    rows = _cluster_rows(all_lines_sorted)
    item_blocks: List[Dict] = []

    for row in rows:
        page = int(row[0].get("page", 0) or 0) if row else 0
        date_x = date_x_by_page.get(page, 250.0)

        desc, nums = _split_columns(row, date_x)
        if not desc or not nums:
            continue

        row_y_candidates = [y_of(l) for l in row]
        row_y = min(row_y_candidates) if row_y_candidates else 0.0

        item_blocks.append(
            {
                "text": " ".join([*desc, *nums]).strip(),
                "description": " ".join(desc).strip(),
                "columns": [n.strip() for n in nums if (n or "").strip()],
                "lines": row,
                "page": page,
                "y": row_y,
            }
        )

    return {
        "raw_text": raw_text,
        "lines": all_lines_sorted,
        "item_blocks": item_blocks,
        "page_count": len(img_paths),
    }
