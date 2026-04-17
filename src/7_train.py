# Pipeline step 7: train classifier (uses 4_manifest, 5_dataset, 6_model).
from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_m = import_module
KneeKLGrayscaleDataset = _m("src.5_dataset").KneeKLGrayscaleDataset
knee_collate_fn = _m("src.5_dataset").knee_collate_fn
build_and_save_manifest = _m("src.4_manifest").build_and_save_manifest
build_resnet18_grayscale = _m("src.6_model").build_resnet18_grayscale
save_eval_reports = _m("src.9_eval_report").save_eval_reports

log = logging.getLogger("knee_oa.train")


def _class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    c = Counter(labels)
    raw = np.array([1.0 / max(c.get(i, 1), 1) for i in range(num_classes)], dtype=np.float32)
    raw = raw * (len(labels) / raw.sum())
    return torch.from_numpy(raw).to(device)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n = 0
    correct = 0
    for xb, yb, _st in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        n += xb.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    stems: list[str] = []
    prob_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb, st_batch in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            prob_chunks.append(prob)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            ps.extend(int(p) for p in pred)
            ys.extend(int(t) for t in yb.numpy().tolist())
            stems.extend(st_batch)
    probs = np.concatenate(prob_chunks, axis=0) if prob_chunks else np.zeros((0, num_classes), dtype=np.float32)
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    acc = float(accuracy_score(ys, ps))
    f1m = float(f1_score(ys, ps, average="macro", zero_division=0))
    f1w = float(f1_score(ys, ps, average="weighted", zero_division=0))
    labels_all = list(range(num_classes))
    names = [f"KL{i}" for i in labels_all]
    report = classification_report(
        ys,
        ps,
        labels=labels_all,
        target_names=names,
        zero_division=0,
    )
    report_dict = classification_report(
        ys,
        ps,
        labels=labels_all,
        target_names=names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "y_true": ys,
        "y_pred": ps,
        "stems": stems,
        "probs": probs,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "report": report,
        "report_dict": report_dict,
    }


def run_training(
    gray_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    seed: int = 42,
    image_size: int = 224,
    num_workers: int = 0,
) -> None:
    out_dir = out_dir.resolve()
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    manifest_path = out_dir / "manifest.json"
    try:
        build_and_save_manifest(
            gray_dir,
            labels_dir,
            manifest_path,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )
    except RuntimeError as e:
        log.error(
            "Manifest build failed (%s). Run: python main.py preprocess --out output/preprocessed",
            e,
        )
        raise
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_recs = manifest["train"]
    val_recs = manifest["val"]
    test_recs = manifest.get("test", [])
    num_classes = int(manifest["num_classes"])

    log.info(
        "train start: epochs=%d batch=%d lr=%g image_size=%d train_n=%d val_n=%d test_n=%d (test held out)",
        epochs,
        batch_size,
        lr,
        image_size,
        len(train_recs),
        len(val_recs),
        len(test_recs),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    train_ds = KneeKLGrayscaleDataset(train_recs, image_size=image_size, augment=True)
    val_ds = KneeKLGrayscaleDataset(val_recs, image_size=image_size, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=knee_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=knee_collate_fn,
    )
    test_loader: DataLoader | None = None
    if test_recs:
        test_ds = KneeKLGrayscaleDataset(test_recs, image_size=image_size, augment=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=knee_collate_fn,
        )

    model = build_resnet18_grayscale(num_classes=num_classes).to(device)
    train_labels = [int(r["kl"]) for r in train_recs]
    weights = _class_weights(train_labels, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history_path = out_dir / "history.csv"
    best_path = out_dir / "best_model.pt"
    cfg_path = out_dir / "train_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "train_frac": train_frac,
                "val_frac": val_frac,
                "test_frac": test_frac,
                "seed": seed,
                "image_size": image_size,
                "gray_dir": str(gray_dir.resolve()),
                "labels_dir": str(labels_dir.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with history_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc = _run_epoch(model, val_loader, device, None, criterion)
        log.info(
            "epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            epochs,
            tr_loss,
            tr_acc,
            va_loss,
            va_acc,
        )
        with history_path.open("a", newline="", encoding="utf-8") as fcsv:
            csv.writer(fcsv).writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}"])

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": va_acc,
                    "num_classes": num_classes,
                    "image_size": image_size,
                },
                best_path,
            )
            log.info("saved best checkpoint val_acc=%.4f -> %s", va_acc, best_path)

    log.info("training finished best_val_acc=%.4f", best_acc)

    if best_path.is_file():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        detail_val = evaluate_detailed(model, val_loader, device, num_classes)
        reports_root = out_dir / "reports"
        rd_v = detail_val["report_dict"]
        kl_names = [f"KL{i}" for i in range(num_classes)]
        meta_val = {
            "split": "validation",
            "split_frac": val_frac,
            "accuracy": detail_val["accuracy"],
            "f1_macro": detail_val["f1_macro"],
            "f1_weighted": detail_val["f1_weighted"],
            "best_checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "best_val_acc_at_save": float(ckpt.get("val_acc", 0.0)),
            "per_class": {k: v for k, v in rd_v.items() if k in kl_names},
            "averages": {k: v for k, v in rd_v.items() if k in ("macro avg", "weighted avg", "micro avg")},
        }
        save_eval_reports(
            reports_root / "validation",
            stems=detail_val["stems"],
            y_true=detail_val["y_true"],
            y_pred=detail_val["y_pred"],
            probs=detail_val["probs"],
            num_classes=num_classes,
            classification_report_text=detail_val["report"],
            meta=meta_val,
        )
        log.info(
            "validation (%.0f%% of data): acc=%.4f f1_macro=%.4f f1_weighted=%.4f",
            val_frac * 100,
            detail_val["accuracy"],
            detail_val["f1_macro"],
            detail_val["f1_weighted"],
        )

        if test_loader is not None:
            detail_te = evaluate_detailed(model, test_loader, device, num_classes)
            rd_t = detail_te["report_dict"]
            meta_te = {
                "split": "test",
                "split_frac": test_frac,
                "accuracy": detail_te["accuracy"],
                "f1_macro": detail_te["f1_macro"],
                "f1_weighted": detail_te["f1_weighted"],
                "best_checkpoint_epoch": int(ckpt.get("epoch", -1)),
                "note": "Held-out test set; not used for checkpoint selection.",
                "per_class": {k: v for k, v in rd_t.items() if k in kl_names},
                "averages": {k: v for k, v in rd_t.items() if k in ("macro avg", "weighted avg", "micro avg")},
            }
            save_eval_reports(
                reports_root / "test",
                stems=detail_te["stems"],
                y_true=detail_te["y_true"],
                y_pred=detail_te["y_pred"],
                probs=detail_te["probs"],
                num_classes=num_classes,
                classification_report_text=detail_te["report"],
                meta=meta_te,
            )
            log.info(
                "test (%.0f%% held out): acc=%.4f f1_macro=%.4f f1_weighted=%.4f",
                test_frac * 100,
                detail_te["accuracy"],
                detail_te["f1_macro"],
                detail_te["f1_weighted"],
            )

        log.info("wrote reports under %s (validation/ and test/)", reports_root)
