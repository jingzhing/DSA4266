from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from pipeline.config import load_config
from pipeline.stages import run_all, run_audit, run_eval, run_infer, run_prepare, run_setup, run_train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified DSA4266 pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_arg(cmd_parser: argparse.ArgumentParser) -> None:
        cmd_parser.add_argument(
            "--config",
            default="configs/pipeline.yaml",
            help="Path to pipeline YAML config",
        )

    setup = subparsers.add_parser("setup", help="Download/validate raw dataset layout")
    add_config_arg(setup)
    setup.add_argument("--force", action="store_true", help="Overwrite existing raw dataset directory")

    prepare = subparsers.add_parser("prepare", help="Build prepared train/val/test dataset")
    add_config_arg(prepare)
    prepare.add_argument("--with-video", action="store_true", help="Enable video frame enrichment")
    prepare.add_argument("--video-url", action="append", default=[], help="Video URL (repeatable)")
    prepare.add_argument("--force", action="store_true", help="Overwrite existing prepared dataset")

    audit = subparsers.add_parser("audit", help="Run raw-data manifest + quality audit")
    add_config_arg(audit)
    audit.add_argument("--force", action="store_true", help="Overwrite existing audit artifacts")

    train = subparsers.add_parser("train", help="Train a model")
    add_config_arg(train)
    train.add_argument("--model", choices=["swin", "efficientnet"], required=True)

    evaluate = subparsers.add_parser("eval", help="Evaluate a trained model")
    add_config_arg(evaluate)
    evaluate.add_argument("--model", choices=["swin", "efficientnet"], required=True)
    evaluate.add_argument("--run-dir", default="", help="Optional run directory override")

    infer = subparsers.add_parser("infer", help="Run batch inference")
    add_config_arg(infer)
    infer.add_argument("--model", choices=["swin", "efficientnet"], required=True)
    infer.add_argument("--input", required=True, help="Input folder containing images")
    infer.add_argument("--run-dir", default="", help="Optional run directory override")

    runall = subparsers.add_parser("run-all", help="Execute setup->audit->prepare->train->eval->infer")
    add_config_arg(runall)
    runall.add_argument("--model", choices=["swin", "efficientnet"], required=True)
    runall.add_argument("--with-video", action="store_true")
    runall.add_argument("--video-url", action="append", default=[], help="Video URL (repeatable)")
    runall.add_argument("--input", default="", help="Optional inference input override")
    runall.add_argument("--skip-setup", action="store_true")
    runall.add_argument("--skip-audit", action="store_true")
    runall.add_argument("--skip-prepare", action="store_true")
    runall.add_argument("--skip-train", action="store_true")
    runall.add_argument("--skip-eval", action="store_true")
    runall.add_argument("--skip-infer", action="store_true")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "setup":
        result = run_setup(cfg, force=args.force)
    elif args.command == "audit":
        result = run_audit(cfg, force=args.force)
    elif args.command == "prepare":
        result = run_prepare(
            cfg,
            with_video=args.with_video,
            video_urls=args.video_url,
            force=args.force,
        )
    elif args.command == "train":
        result = run_train(cfg, model_name=args.model)
    elif args.command == "eval":
        run_dir = Path(args.run_dir) if args.run_dir else None
        result = run_eval(cfg, model_name=args.model, run_dir=run_dir)
    elif args.command == "infer":
        run_dir = Path(args.run_dir) if args.run_dir else None
        result = run_infer(cfg, model_name=args.model, input_dir=Path(args.input), run_dir=run_dir)
    elif args.command == "run-all":
        infer_input = Path(args.input) if args.input else None
        result = run_all(
            cfg=cfg,
            model_name=args.model,
            with_video=args.with_video,
            video_urls=args.video_url,
            infer_input=infer_input,
            skip_setup=args.skip_setup,
            skip_audit=args.skip_audit,
            skip_prepare=args.skip_prepare,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            skip_infer=args.skip_infer,
        )
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
