# Pipeline step 10: compare multiple trained runs (metrics + charts).
from __future__ import annotations

import csv
import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_m = import_module
KneeKLGrayscaleDataset = _m("src.5_dataset").KneeKLGrayscaleDataset
knee_collate_fn = _m("src.5_dataset").knee_collate_fn
build_knee_classifier = _m("src.6_model").build_knee_classifier
normalize_architecture = _m("src.6_model").normalize_architecture
evaluate_detailed = _m("src.7_train").evaluate_detailed

log = logging.getLogger("knee_oa.compare")


def _split_fingerprint(manifest: dict[str, Any], split_key: str) -> list[tuple[str, int]]:
    recs = manifest.get(split_key) or []
    return sorted((str(r["stem"]), int(r["kl"])) for r in recs)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "train_config.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _history_val_accs(history_csv: Path) -> tuple[list[int], list[float]]:
    if not history_csv.is_file():
        return [], []
    epochs: list[int] = []
    accs: list[float] = []
    with history_csv.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            accs.append(float(row["val_acc"]))
    return epochs, accs


def run_compare(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    split: str = "test",
    batch_size: int = 16,
    num_workers: int = 0,
) -> None:
    if len(run_dirs) < 2:
        raise ValueError("compare needs at least two --runs directories.")

    run_dirs = [d.resolve() for d in run_dirs]
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    split = split.lower().strip()
    if split in ("val", "validation"):
        split_key = "val"
        split_label = "validation"
    elif split == "test":
        split_key = "test"
        split_label = "test"
    else:
        raise ValueError("split must be 'val' or 'test'")

    ref_manifest_path = run_dirs[0] / "manifest.json"
    if not ref_manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {ref_manifest_path}")
    ref_manifest = json.loads(ref_manifest_path.read_text(encoding="utf-8"))
    ref_fp = _split_fingerprint(ref_manifest, split_key)
    if not ref_fp and split_key == "test":
        log.warning("reference manifest has no test split; using val for comparison.")
        split_key = "val"
        split_label = "validation"
        ref_fp = _split_fingerprint(ref_manifest, split_key)
    if not ref_fp:
        raise ValueError("Manifest has no samples in the requested split.")

    num_classes = int(ref_manifest["num_classes"])
    results: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        man_path = run_dir / "manifest.json"
        ckpt_path = run_dir / "best_model.pt"
        if not man_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {man_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        fp = _split_fingerprint(manifest, split_key)
        if fp != ref_fp:
            raise ValueError(
                f"Split '{split_label}' differs from reference run {run_dirs[0].name!r} "
                f"in {run_dir.name!r} (stems/KL must match for a fair compare). "
                "Train with the same data, seed, and split fractions, or use the same manifest."
            )
        if int(manifest["num_classes"]) != num_classes:
            raise ValueError(f"num_classes mismatch in {run_dir}")

        cfg = _load_run_config(run_dir)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_name = normalize_architecture(str(ckpt.get("model") or cfg.get("model") or "resnet50"))
        image_size = int(ckpt.get("image_size") or cfg.get("image_size") or 224)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_knee_classifier(model_name, num_classes=num_classes).to(device)
        model.load_state_dict(ckpt["model_state"])

        recs = manifest[split_key]
        eval_ds = KneeKLGrayscaleDataset(recs, image_size=image_size, augment=False)
        loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=knee_collate_fn,
        )
        detail = evaluate_detailed(model, loader, device, num_classes)

        hist_path = run_dir / "history.csv"
        ep, va = _history_val_accs(hist_path)
        best_val = max(va) if va else float(ckpt.get("val_acc", 0.0))

        label = f"{model_name} ({run_dir.name})"
        results.append(
            {
                "run_dir": str(run_dir),
                "label": label,
                "model": model_name,
                "accuracy": detail["accuracy"],
                "f1_macro": detail["f1_macro"],
                "f1_weighted": detail["f1_weighted"],
                "best_val_acc_training": float(best_val),
            }
        )
        curve_rows.append({"label": label, "epochs": ep, "val_accs": va})
        log.info(
            "%s — %s acc=%.4f f1_macro=%.4f f1_weighted=%.4f",
            run_dir.name,
            model_name,
            detail["accuracy"],
            detail["f1_macro"],
            detail["f1_weighted"],
        )

    summary = {
        "split": split_label,
        "num_samples": len(ref_fp),
        "num_classes": num_classes,
        "runs": results,
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out_dir / "comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_dir",
                "model",
                "label",
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "best_val_acc_training",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["run_dir"],
                    r["model"],
                    r["label"],
                    f"{r['accuracy']:.6f}",
                    f"{r['f1_macro']:.6f}",
                    f"{r['f1_weighted']:.6f}",
                    f"{r['best_val_acc_training']:.6f}",
                ]
            )

    _save_compare_figures(out_dir / "figures", results, curve_rows, split_label)
    log.info("comparison artifacts under %s", out_dir)


def _save_compare_figures(
    fig_dir: Path,
    results: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    split_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x - w, [r["accuracy"] for r in results], width=w, label="Accuracy")
    ax.bar(x, [r["f1_macro"] for r in results], width=w, label="F1 macro")
    ax.bar(x + w, [r["f1_weighted"] for r in results], width=w, label="F1 weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Metrics on {split_label} split (same samples across runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for cr in curve_rows:
        ep = cr["epochs"]
        va = cr["val_accs"]
        if ep and va:
            ax2.plot(ep, va, marker="o", markersize=2, label=cr["label"])
    if any(cr["epochs"] for cr in curve_rows):
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation accuracy")
        ax2.set_title("Training validation accuracy (per run)")
        ax2.legend()
        ax2.set_ylim(0, 1.02)
        ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "val_acc_curves.png", dpi=150)
    plt.close(fig2)
