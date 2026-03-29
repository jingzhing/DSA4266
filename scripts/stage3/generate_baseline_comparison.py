from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_model_name(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "efficientnet" in name:
        return "efficientnet"
    if "swin" in name:
        return "swin"
    return "unknown"


def _run_record(run_dir: Path) -> Dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    train = _load_json(run_dir / "train_summary.json")
    eval_summary = _load_json(run_dir / "eval_summary.json")
    return {
        "model": _infer_model_name(run_dir),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_path": str(train.get("checkpoint_path", "")),
        "best_threshold": train.get("best_threshold"),
        "best_val_balanced_accuracy": train.get("best_val_balanced_accuracy"),
        "best_val_auc": train.get("best_val_auc"),
        "test_accuracy": metrics.get("accuracy"),
        "test_balanced_accuracy": metrics.get("balanced_accuracy"),
        "test_precision": metrics.get("precision"),
        "test_recall": metrics.get("recall"),
        "test_f1": metrics.get("f1"),
        "test_roc_auc": metrics.get("roc_auc"),
        "eval_prediction_count": eval_summary.get("prediction_count"),
    }


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(records: List[Dict[str, Any]], generated_at_utc: str) -> str:
    lines = [
        "# Phase-3 Baseline Model Comparison",
        "",
        f"- Generated at (UTC): `{generated_at_utc}`",
        "",
        "## Runs",
        "",
    ]
    for record in records:
        lines.append(f"- `{record['model']}`: `{record['run_dir']}`")
    lines.extend(
        [
            "",
            "## Validation (Best Epoch/Threshold)",
            "",
            "| Model | Best Threshold | Best Val Balanced Acc | Best Val AUC |",
            "|---|---:|---:|---:|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['model']} | "
            f"{_format(record['best_threshold'])} | "
            f"{_format(record['best_val_balanced_accuracy'])} | "
            f"{_format(record['best_val_auc'])} |"
        )

    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            "| Model | Accuracy | Balanced Acc | Precision | Recall | F1 | ROC AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['model']} | "
            f"{_format(record['test_accuracy'])} | "
            f"{_format(record['test_balanced_accuracy'])} | "
            f"{_format(record['test_precision'])} | "
            f"{_format(record['test_recall'])} | "
            f"{_format(record['test_f1'])} | "
            f"{_format(record['test_roc_auc'])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Metrics are parsed from each run's `train_summary.json`, `metrics.json`, and `eval_summary.json`.",
            "- This report compares Phase-3 baseline runs on the deterministic subset dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one consolidated baseline comparison report.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory path (repeatable)")
    parser.add_argument(
        "--out-md",
        default="docs/PHASE3_BASELINE_COMPARISON.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--out-json",
        default="docs/PHASE3_BASELINE_COMPARISON.json",
        help="Output machine-readable JSON path",
    )
    args = parser.parse_args()

    run_dirs = [Path(path).resolve() for path in args.run_dir]
    missing = [str(path) for path in run_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run directory not found: {missing}")

    records = [_run_record(path) for path in run_dirs]
    generated_at_utc = datetime.now(timezone.utc).isoformat()

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at_utc,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out_md = Path(args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown(records, generated_at_utc), encoding="utf-8")

    print(json.dumps({"out_md": str(out_md), "out_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
