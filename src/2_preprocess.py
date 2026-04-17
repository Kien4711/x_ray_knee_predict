# Pipeline step 2: load images + YOLO labels; grayscale, knee crop, preview.
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

_pre_log = logging.getLogger("knee_oa.preprocess")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


# BGR — mau khung theo muc KL (class YOLO 0..4 -> KL0..KL4)
KL_BOX_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (200, 200, 200),  # KL0
    1: (0, 200, 100),  # KL1
    2: (0, 180, 255),  # KL2
    3: (0, 0, 255),  # KL3
    4: (255, 0, 255),  # KL4
}


def kl_label_for_class(class_id: int) -> str:
    if class_id in KL_BOX_COLORS_BGR:
        return f"KL{class_id}"
    return f"c{class_id}"


def read_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def bgr_to_gray_luma(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr.copy()
    if img_bgr.shape[2] == 1:
        return img_bgr[:, :, 0].copy()
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def parse_yolo_label_file(label_path: Path) -> list[YoloBox]:
    if not label_path.is_file():
        return []
    rows: list[YoloBox] = []
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        rows.append(YoloBox(class_id=cid, cx=cx, cy=cy, w=w, h=h))
    return rows


def yolo_to_xyxy(box: YoloBox, width: int, height: int) -> tuple[int, int, int, int]:
    px_w = box.w * width
    px_h = box.h * height
    cx = box.cx * width
    cy = box.cy * height
    x1 = int(round(cx - px_w / 2))
    y1 = int(round(cy - px_h / 2))
    x2 = int(round(cx + px_w / 2))
    y2 = int(round(cy + px_h / 2))
    return x1, y1, x2, y2


def union_xyxy(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return x1, y1, x2, y2


def pad_xyxy(
    rect: tuple[int, int, int, int],
    pad_ratio: float,
    bounds: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    pad_x = int(round(w * pad_ratio))
    pad_y = int(round(h * pad_ratio))
    bw, bh = bounds
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(bw, x2 + pad_x)
    ny2 = min(bh, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_kl_boxes_bgr(
    canvas_bgr: np.ndarray,
    boxes_yolo: Iterable[YoloBox],
    offset_x: int = 0,
    offset_y: int = 0,
    img_width: int | None = None,
    img_height: int | None = None,
) -> np.ndarray:
    _, w = canvas_bgr.shape[:2]
    iw = img_width if img_width is not None else w
    ih = img_height if img_height is not None else canvas_bgr.shape[0]
    out = canvas_bgr
    for b in boxes_yolo:
        x1, y1, x2, y2 = yolo_to_xyxy(b, iw, ih)
        x1 -= offset_x
        y1 -= offset_y
        x2 -= offset_x
        y2 -= offset_y
        color = KL_BOX_COLORS_BGR.get(b.class_id, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
        tag = kl_label_for_class(b.class_id)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(0, y1 - 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(
            out,
            tag,
            (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def preprocess_pair(
    image_path: Path,
    label_path: Path | None,
    *,
    pad_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (gray_crop_1ch, preview_bgr_3ch).
    gray_crop: uint8 single channel, knee ROI (union of YOLO boxes + padding).
    preview: BGR overlay for visualization (colored KL boxes on grayscale).
    """
    bgr = read_image_bgr(image_path)
    gray_full = bgr_to_gray_luma(bgr)
    ih, iw = gray_full.shape[:2]

    boxes = parse_yolo_label_file(label_path) if label_path else []
    pixel_boxes = [yolo_to_xyxy(b, iw, ih) for b in boxes]

    if pixel_boxes:
        u = union_xyxy(pixel_boxes)
        assert u is not None
        crop = pad_xyxy(u, pad_ratio, (iw, ih))
        x1, y1, x2, y2 = crop
    else:
        x1, y1, x2, y2 = 0, 0, iw, ih

    gray_crop = gray_full[y1:y2, x1:x2].copy()
    preview = gray_to_bgr(gray_crop)
    draw_kl_boxes_bgr(
        preview,
        boxes,
        offset_x=x1,
        offset_y=y1,
        img_width=iw,
        img_height=ih,
    )
    return gray_crop, preview


def write_gray_png(path: Path, gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    buf.tofile(str(path))


def write_bgr_png(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    buf.tofile(str(path))


def default_label_path(images_dir: Path, labels_dir: Path, image_path: Path) -> Path:
    stem = image_path.stem
    return labels_dir / f"{stem}.txt"


def run_batch(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    *,
    pad_ratio: float = 0.08,
    pattern: str = "*.jpg",
    logger: logging.Logger | None = None,
    max_files: int | None = None,
) -> None:
    log = logger or _pre_log
    gray_dir = out_dir / "gray_crop"
    prev_dir = out_dir / "preview"
    images = sorted(images_dir.glob(pattern))
    if max_files is not None:
        images = images[: max(0, max_files)]
    log.info(
        "preprocess batch: images_dir=%s labels_dir=%s out=%s n_images=%d pad=%.4f pattern=%s max_files=%s",
        images_dir,
        labels_dir,
        out_dir,
        len(images),
        pad_ratio,
        pattern,
        max_files,
    )
    ok = 0
    failed = 0
    for img_path in images:
        stem = img_path.stem
        try:
            lbl_path = default_label_path(images_dir, labels_dir, img_path)
            label_file: Path | None = lbl_path if lbl_path.is_file() else None
            if label_file is None:
                log.warning("no label for %s (expected %s); using full image crop", stem, lbl_path)
            gray, preview = preprocess_pair(img_path, label_file, pad_ratio=pad_ratio)
            write_gray_png(gray_dir / f"{stem}.png", gray)
            write_bgr_png(prev_dir / f"{stem}.png", preview)
            ok += 1
            log.debug("ok %s", stem)
        except Exception:
            failed += 1
            log.exception("failed %s", img_path)
    log.info("preprocess finished: ok=%d failed=%d total=%d", ok, failed, len(images))
