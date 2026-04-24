# Pipeline step 8: evaluate checkpoint (uses 5_dataset, 6_model, 7_train.evaluate_detailed).
from __future__ import annotations

import json
import logging
from importlib import import_module
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_m = import_module
KneeKLGrayscaleDataset = _m("src.5_dataset").KneeKLGrayscaleDataset
knee_collate_fn = _m("src.5_dataset").knee_collate_fn
build_knee_classifier = _m("src.6_model").build_knee_classifier
normalize_architecture = _m("src.6_model").normalize_architecture
evaluate_detailed = _m("src.7_train").evaluate_detailed
save_eval_reports = _m("src.9_eval_report").save_eval_reports

log = logging.getLogger("knee_oa.eval")


def run_evaluate(
    manifest_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    *,
    split: str = "test",
    batch_size: int = 16,
    num_workers: int = 0,
    model_name: str | None = None,
) -> None:
    manifest_path = manifest_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = split.lower().strip()
    if split not in ("val", "validation", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if split in ("val", "validation"):
        split_key = "val"
        split_label = "validation"
    else:
        split_key = "test"
        split_label = "test"

    recs = manifest.get(split_key)
    if not recs and split_key == "test":
        log.warning("manifest has no 'test' split; falling back to 'val'")
        recs = manifest["val"]
        split_label = "validation"
    elif not recs:
        recs = manifest["val"]
    num_classes = int(manifest["num_classes"])
    image_size = 224
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "image_size" in ckpt:
        image_size = int(ckpt["image_size"])
    raw = model_name if model_name is not None else ckpt.get("model") or "resnet50"
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    arch = normalize_architecture(str(raw))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_knee_classifier(arch, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])

    eval_ds = KneeKLGrayscaleDataset(recs, image_size=image_size, augment=False)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=knee_collate_fn,
    )

    detail = evaluate_detailed(model, eval_loader, device, num_classes)
    log.info(
        "%s accuracy=%.4f f1_macro=%.4f f1_weighted=%.4f",
        split_label,
        detail["accuracy"],
        detail["f1_macro"],
        detail["f1_weighted"],
    )
    reports_dir = out_dir / "reports" / split_label
    rd = detail["report_dict"]
    kl_names = [f"KL{i}" for i in range(num_classes)]
    meta = {
        "split": split_label,
        "accuracy": detail["accuracy"],
        "f1_macro": detail["f1_macro"],
        "f1_weighted": detail["f1_weighted"],
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "per_class": {k: v for k, v in rd.items() if k in kl_names},
        "averages": {k: v for k, v in rd.items() if k in ("macro avg", "weighted avg", "micro avg")},
    }
    save_eval_reports(
        reports_dir,
        stems=detail["stems"],
        y_true=detail["y_true"],
        y_pred=detail["y_pred"],
        probs=detail["probs"],
        num_classes=num_classes,
        classification_report_text=detail["report"],
        meta=meta,
    )
    log.info("wrote reports under %s", reports_dir)
