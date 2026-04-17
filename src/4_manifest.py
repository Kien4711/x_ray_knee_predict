# Pipeline step 4: build sample list + train/val/test split (uses 3_labels).
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

infer_image_kl_from_label_file = import_module("src.3_labels").infer_image_kl_from_label_file

log = logging.getLogger("knee_oa.manifest")


@dataclass
class SampleRecord:
    stem: str
    gray_png: str
    label_txt: str
    kl: int


def collect_samples(
    gray_dir: Path,
    labels_dir: Path,
) -> list[SampleRecord]:
    gray_dir = gray_dir.resolve()
    labels_dir = labels_dir.resolve()
    records: list[SampleRecord] = []
    for png in sorted(gray_dir.glob("*.png")):
        stem = png.stem
        lbl = labels_dir / f"{stem}.txt"
        kl = infer_image_kl_from_label_file(lbl)
        if kl is None:
            log.warning("skip %s: no KL from label %s", stem, lbl)
            continue
        records.append(
            SampleRecord(
                stem=stem,
                gray_png=str(png.resolve()),
                label_txt=str(lbl.resolve()),
                kl=int(kl),
            )
        )
    log.info("collected %d samples from %s", len(records), gray_dir)
    return records


def _can_stratify(labels: list[int]) -> bool:
    from collections import Counter

    c = Counter(labels)
    return all(v >= 2 for v in c.values()) and len(c) >= 1


def stratified_split_train_val_test(
    labels: list[int],
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-5:
        raise ValueError(f"train_frac + val_frac + test_frac must be 1, got {train_frac + val_frac + test_frac}")
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise RuntimeError("pip install scikit-learn for splits") from e

    indices = list(range(len(labels)))
    strat_all = labels if _can_stratify(labels) else None
    idx_trva, idx_te = _train_test_split_safe(
        indices,
        labels,
        test_size=test_frac,
        random_state=seed,
        stratify_labels=strat_all,
        log=log,
        step="train+val vs test",
    )
    sub_labels = [labels[i] for i in idx_trva]
    strat_trva = sub_labels if _can_stratify(sub_labels) else None
    rel_val = val_frac / (train_frac + val_frac)
    idx_tr, idx_va = _train_test_split_safe(
        idx_trva,
        sub_labels,
        test_size=rel_val,
        random_state=seed,
        stratify_labels=strat_trva,
        log=log,
        step="train vs val",
    )
    return sorted(idx_tr), sorted(idx_va), sorted(idx_te)


def _train_test_split_safe(
    indices: list[int],
    labels_for_stratify: list[int],
    *,
    test_size: float,
    random_state: int,
    stratify_labels: list[int] | None,
    log: logging.Logger,
    step: str,
) -> tuple[list[int], list[int]]:
    from sklearn.model_selection import train_test_split

    if stratify_labels is None:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError:
        log.warning(
            "stratified split not possible for %s (often small n or rare classes); using random split",
            step,
        )
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )


def build_and_save_manifest(
    gray_dir: Path,
    labels_dir: Path,
    out_json: Path,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    seed: int,
) -> dict[str, Any]:
    samples = collect_samples(gray_dir, labels_dir)
    if not samples:
        raise RuntimeError("No samples: check gray_dir and labels_dir")
    label_list = [s.kl for s in samples]
    tr_idx, va_idx, te_idx = stratified_split_train_val_test(
        label_list,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    train_recs = [samples[i] for i in tr_idx]
    val_recs = [samples[i] for i in va_idx]
    test_recs = [samples[i] for i in te_idx]
    n = len(samples)
    payload = {
        "seed": seed,
        "split": {
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "train_n": len(train_recs),
            "val_n": len(val_recs),
            "test_n": len(test_recs),
            "total_n": n,
        },
        "num_classes": 5,
        "train": [asdict(s) for s in train_recs],
        "val": [asdict(s) for s in val_recs],
        "test": [asdict(s) for s in test_recs],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "manifest: train=%d val=%d test=%d (%.0f-%.0f-%.0f) -> %s",
        len(train_recs),
        len(val_recs),
        len(test_recs),
        train_frac * 10,
        val_frac * 10,
        test_frac * 10,
        out_json,
    )
    return payload
