from __future__ import annotations

import argparse
import logging
from importlib import import_module
from pathlib import Path

_m = import_module
configure_knee_oa_logging = _m("src.1_run_logging").configure_knee_oa_logging
_pre = _m("src.2_preprocess")
preprocess_pair = _pre.preprocess_pair
run_batch = _pre.run_batch
write_bgr_png = _pre.write_bgr_png
write_gray_png = _pre.write_gray_png
run_training = _m("src.7_train").run_training
run_evaluate = _m("src.8_eval_run").run_evaluate
run_compare = _m("src.10_compare").run_compare
ARCHITECTURES = _m("src.6_model").ARCHITECTURES


def main() -> None:
    parser = argparse.ArgumentParser(description="Knee OA: preprocess, train, evaluate.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_batch = sub.add_parser("preprocess", help="Batch: Images_E + Labels_E")
    p_batch.add_argument("--images", type=Path, default=Path("data/Images_E"))
    p_batch.add_argument("--labels", type=Path, default=Path("data/Labels_E"))
    p_batch.add_argument("--out", type=Path, default=Path("output/preprocessed"))
    p_batch.add_argument("--pad", type=float, default=0.08)
    p_batch.add_argument("--pattern", type=str, default="*.jpg")
    p_batch.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of images (dev/smoke; default: all)",
    )

    p_one = sub.add_parser("preprocess-one", help="Single image + label")
    p_one.add_argument("--image", type=Path, required=True)
    p_one.add_argument("--label", type=Path, required=True)
    p_one.add_argument("--out", type=Path, default=Path("output/preprocessed"))
    p_one.add_argument("--pad", type=float, default=0.08)

    p_train = sub.add_parser("train", help="Train KL classifier on gray_crop + YOLO labels")
    p_train.add_argument("--gray-dir", type=Path, default=Path("output/preprocessed/gray_crop"))
    p_train.add_argument("--labels-dir", type=Path, default=Path("data/Labels_E"))
    p_train.add_argument("--out", type=Path, default=Path("output/run_latest"))
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--train-frac", type=float, default=0.7, help="Train ratio (default 0.7 = 7 in 7-2-1)")
    p_train.add_argument("--val-frac", type=float, default=0.2, help="Validation ratio (default 0.2)")
    p_train.add_argument("--test-frac", type=float, default=0.1, help="Test holdout ratio (default 0.1)")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--image-size", type=int, default=224)
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=ARCHITECTURES,
        help="Backbone: resnet50, efficientnet_b0, mobilenet_v2",
    )

    p_ev = sub.add_parser("evaluate", help="Evaluate checkpoint on manifest val or test split")
    p_ev.add_argument("--manifest", type=Path, required=True)
    p_ev.add_argument("--checkpoint", type=Path, required=True)
    p_ev.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("test", "val"),
        help="Which manifest split to score (default: test holdout)",
    )
    p_ev.add_argument("--out", type=Path, default=Path("output/eval_out"))
    p_ev.add_argument("--batch-size", type=int, default=16)
    p_ev.add_argument("--num-workers", type=int, default=0)
    p_ev.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="ARCH",
        help=f"Optional backbone override ({', '.join(ARCHITECTURES)}); default: from checkpoint",
    )

    p_cmp = sub.add_parser(
        "compare",
        help="Compare >=2 trained runs on the same split (metrics + charts); manifests must match",
    )
    p_cmp.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Training output dirs (each with manifest.json, best_model.pt, history.csv)",
    )
    p_cmp.add_argument("--out", type=Path, default=Path("output/compare_latest"))
    p_cmp.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("test", "val"),
        help="Which manifest split to score (default: test)",
    )
    p_cmp.add_argument("--batch-size", type=int, default=16)
    p_cmp.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    if args.command == "preprocess":
        out = args.out.resolve()
        log_dir = out / "logs"
        configure_knee_oa_logging(log_dir / "preprocess.log")
        run_batch(
            args.images.resolve(),
            args.labels.resolve(),
            out,
            pad_ratio=args.pad,
            pattern=args.pattern,
            max_files=args.limit,
        )
        logging.getLogger("knee_oa.main").info("artifacts under %s (see logs/preprocess.log)", out)
        return

    if args.command == "preprocess-one":
        out = args.out.resolve()
        log_dir = out / "logs"
        configure_knee_oa_logging(log_dir / "preprocess_one.log")
        log = logging.getLogger("knee_oa.main")
        stem = args.image.stem
        log.info("preprocess-one: image=%s label=%s out=%s", args.image, args.label, out)
        out.mkdir(parents=True, exist_ok=True)
        gray, preview = preprocess_pair(
            args.image.resolve(),
            args.label.resolve(),
            pad_ratio=args.pad,
        )
        write_gray_png((out / "gray_crop" / f"{stem}.png").resolve(), gray)
        write_bgr_png((out / "preview" / f"{stem}.png").resolve(), preview)
        log.info("wrote gray_crop/%s.png and preview/", stem)
        return

    if args.command == "train":
        out = args.out.resolve()
        log_dir = out / "logs"
        configure_knee_oa_logging(log_dir / "train.log")
        run_training(
            args.gray_dir.resolve(),
            args.labels_dir.resolve(),
            out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            image_size=args.image_size,
            num_workers=args.num_workers,
            model_name=args.model,
        )
        logging.getLogger("knee_oa.main").info("run output: %s (see logs/train.log)", out)
        return

    if args.command == "evaluate":
        if args.model is not None and args.model not in ARCHITECTURES:
            parser.error(f"--model must be one of: {', '.join(ARCHITECTURES)}")
        out = args.out.resolve()
        log_dir = out / "logs"
        configure_knee_oa_logging(log_dir / "evaluate.log")
        run_evaluate(
            args.manifest.resolve(),
            args.checkpoint.resolve(),
            out,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_name=args.model,
        )
        logging.getLogger("knee_oa.main").info("eval output: %s", out)
        return

    if args.command == "compare":
        out = args.out.resolve()
        log_dir = out / "logs"
        configure_knee_oa_logging(log_dir / "compare.log")
        run_compare(
            [p.resolve() for p in args.runs],
            out,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        logging.getLogger("knee_oa.main").info("compare output: %s", out)
        return


if __name__ == "__main__":
    main()
