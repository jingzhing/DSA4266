import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(cmd, name, workdir):
    print("\n" + "=" * 88)
    print(f"STARTING: {name}")
    print("COMMAND :", " ".join(cmd))
    print("=" * 88)
    t0 = time.time()

    process = subprocess.run(cmd, cwd=workdir)
    dt = time.time() - t0

    if process.returncode != 0:
        print(f"\nFAILED: {name} | exit_code={process.returncode} | elapsed={dt/60:.2f} min")
        raise SystemExit(process.returncode)

    print(f"\nDONE: {name} | elapsed={dt/60:.2f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=".", help="Folder containing train_single.py, test_single.py, ensemble_posthoc.py")
    parser.add_argument("--skip-train", action="store_true", help="Skip both training steps")
    parser.add_argument("--skip-test", action="store_true", help="Skip both solo test steps")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble step")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for subprocess calls. Defaults to current interpreter.",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    py = args.python

    required = ["train_single.py", "test_single.py", "ensemble_posthoc.py"]
    missing = [name for name in required if not (workdir / name).exists()]
    if missing:
        print("Missing required files in workdir:")
        for name in missing:
            print(" -", name)
        raise SystemExit(2)

    steps = []

    if not args.skip_train:
        steps.extend([
            ([py, "train_single.py", "--model", "swin"], "Train Swin"),
            ([py, "train_single.py", "--model", "efficientnet"], "Train EfficientNet"),
        ])

    if not args.skip_test:
        steps.extend([
            ([py, "test_single.py", "--model", "swin"], "Test Swin"),
            ([py, "test_single.py", "--model", "efficientnet"], "Test EfficientNet"),
        ])

    if not args.skip_ensemble:
        steps.append(([py, "ensemble_posthoc.py"], "Run Ensemble"))

    if not steps:
        print("Nothing to run. Remove one of the --skip-* flags.")
        return

    print("Working directory:", workdir)
    print("Python executable:", py)

    total_start = time.time()
    for cmd, name in steps:
        run_step(cmd, name, workdir)

    total_dt = time.time() - total_start
    print("\n" + "#" * 88)
    print(f"PIPELINE COMPLETE | total_elapsed={total_dt/3600:.2f} hours")
    print("#" * 88)


if __name__ == "__main__":
    main()
