# Pipeline step 3: infer image-level KL from YOLO file (uses 2_preprocess).
from __future__ import annotations

from importlib import import_module
from pathlib import Path

parse_yolo_label_file = import_module("src.2_preprocess").parse_yolo_label_file

# Image-level KL (heuristic): max YOLO class id in 0..4 in that label file.
KL_CLASS_MIN = 0
KL_CLASS_MAX = 4


def infer_image_kl_from_label_file(label_path: Path) -> int | None:
    if not label_path.is_file():
        return None
    boxes = parse_yolo_label_file(label_path)
    if not boxes:
        return None
    ids = [b.class_id for b in boxes if KL_CLASS_MIN <= b.class_id <= KL_CLASS_MAX]
    if not ids:
        return None
    return max(ids)
