# Pipeline step 9: export evaluation artifacts (confusion matrix, CSV, JSON summary).
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_eval_reports(
    reports_dir: Path,
    *,
    stems: list[str],
    y_true: list[int],
    y_pred: list[int],
    probs: np.ndarray,
    num_classes: int,
    classification_report_text: str,
    meta: dict[str, Any],
) -> None:
    reports_dir = reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    rep_txt = reports_dir / "classification_report.txt"
    rep_txt.write_text(classification_report_text, encoding="utf-8")

    _write_predictions_csv(reports_dir / "predictions.csv", stems, y_true, y_pred, probs, num_classes)
    _write_confusion_csv_png(reports_dir, y_true, y_pred, num_classes)

    summary = _json_safe(
        {
            **meta,
            "num_samples": len(y_true),
            "num_classes": num_classes,
            "confusion_matrix_csv": "confusion_matrix.csv",
            "confusion_matrix_png": "confusion_matrix.png",
            "predictions_csv": "predictions.csv",
            "classification_report_txt": "classification_report.txt",
        }
    )
    (reports_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_predictions_csv(
    path: Path,
    stems: list[str],
    y_true: list[int],
    y_pred: list[int],
    probs: np.ndarray,
    num_classes: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["stem", "y_true", "y_pred", "correct"] + [f"prob_KL{i}" for i in range(num_classes)]
        w.writerow(header)
        for i, stem in enumerate(stems):
            row = [stem, y_true[i], y_pred[i], int(y_true[i] == y_pred[i])]
            row.extend(f"{float(p):.6f}" for p in probs[i])
            w.writerow(row)


def _write_confusion_csv_png(
    reports_dir: Path,
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    csv_path = reports_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"pred_KL{j}" for j in range(num_classes)])
        for i in range(num_classes):
            w.writerow([f"true_KL{i}"] + [int(cm[i, j]) for j in range(num_classes)])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=[f"KL{i}" for i in range(num_classes)],
            yticklabels=[f"KL{i}" for i in range(num_classes)],
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
