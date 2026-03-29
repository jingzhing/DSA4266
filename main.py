from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _kaggle_token_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def _fmt_cmd(cmd: Iterable[str]) -> str:
    return " ".join(cmd)


def _run_step(name: str, cmd: List[str]) -> None:
    print(f"[main.py] START {name}: {_fmt_cmd(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[main.py] DONE  {name}", flush=True)


def _run_step_with_log(name: str, cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[main.py] START {name}: {_fmt_cmd(cmd)}", flush=True)
    print(f"[main.py] LOG   {name}: {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

    print(f"[main.py] DONE  {name}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-command runner for setup->audit->prepare->train (SWIN by default)."
    )
    parser.add_argument(
        "--setup-config",
        default="configs/pipeline.yaml",
        help="Config used for setup/audit/prepare.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/pipeline_full_swin_optimized.yaml",
        help="Config used for train.",
    )
    parser.add_argument(
        "--model",
        default="swin",
        choices=["swin", "efficientnet"],
        help="Model to train.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies from requirements.txt before pipeline stages.",
    )
    parser.add_argument("--skip-setup", action="store_true", help="Skip setup stage.")
    parser.add_argument("--skip-audit", action="store_true", help="Skip audit stage.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare stage.")
    parser.add_argument(
        "--no-force-audit",
        action="store_true",
        help="Do not pass --force to audit.",
    )
    parser.add_argument(
        "--no-force-prepare",
        action="store_true",
        help="Do not pass --force to prepare.",
    )
    parser.add_argument(
        "--with-video",
        action="store_true",
        help="Pass --with-video to prepare (in addition to config settings).",
    )
    parser.add_argument(
        "--video-url",
        action="append",
        default=[],
        help="Repeatable video URL to pass into prepare.",
    )
    parser.add_argument(
        "--train-log",
        default="outputs/logs/main_swin_full_train.log",
        help="Path for combined stdout/stderr log for the train step.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print steps without executing.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = _repo_root()
    setup_cfg = (root / args.setup_config).resolve()
    train_cfg = (root / args.train_config).resolve()
    train_log = (root / args.train_log).resolve()
    _require_path(setup_cfg, "Setup config")
    _require_path(train_cfg, "Train config")
    requirements_path = (root / "requirements.txt").resolve()
    if args.install_deps:
        _require_path(requirements_path, "requirements.txt")

    python_exe = sys.executable
    steps: List[tuple[str, List[str], bool]] = []

    if args.install_deps:
        steps.append(
            (
                "install-deps",
                [python_exe, "-m", "pip", "install", "-r", str(requirements_path)],
                False,
            )
        )

    if not args.skip_setup:
        token_path = _kaggle_token_path()
        if not token_path.exists():
            raise FileNotFoundError(
                "Kaggle token missing. Expected at "
                f"{token_path}. Create the file before running setup."
            )
        steps.append(
            (
                "setup",
                [python_exe, "-m", "pipeline.cli", "setup", "--config", str(setup_cfg)],
                False,
            )
        )

    if not args.skip_audit:
        audit_cmd = [python_exe, "-m", "pipeline.cli", "audit", "--config", str(setup_cfg)]
        if not args.no_force_audit:
            audit_cmd.append("--force")
        steps.append(("audit", audit_cmd, False))


    # Always skip video enrichment, only use Kaggle dataset and augmentation for fake class
    if not args.skip_prepare:
        prepare_cmd = [python_exe, "-m", "pipeline.cli", "prepare", "--config", str(setup_cfg), "--force"]
        # Explicitly do NOT add --with-video or --video-url
        steps.append(("prepare", prepare_cmd, False))


    # Enable batch logging for SWIN training by setting env variable or config
    train_cmd = [
        python_exe,
        "-u",
        "-m",
        "pipeline.cli",
        "train",
        "--config",
        str(train_cfg),
        "--model",
        args.model,
    ]
    # Optionally, you can add a debug flag or ensure the config enables batch logging
    steps.append(("train", train_cmd, True))

    if args.dry_run:
        for name, cmd, _logged in steps:
            print(f"[main.py][dry-run] {name}: {_fmt_cmd(cmd)}")
        return 0

    for name, cmd, logged in steps:
        if logged:
            _run_step_with_log(name, cmd, train_log)
        else:
            _run_step(name, cmd)

    print("[main.py] Pipeline run completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
