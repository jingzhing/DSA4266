from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import brier_score_loss

from pipeline.metrics import balanced_accuracy


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def write_train_research_artifacts(
    run_dir: Path,
    train_summary: Dict[str, Any],
    cfg: Dict[str, Any],
    model_name: str,
) -> None:
    history = train_summary.get("history", [])
    if history and isinstance(history, list):
        history_path = run_dir / "train_epoch_history.jsonl"
        _ensure_dir(history_path)
        with history_path.open("w", encoding="utf-8") as handle:
            for row in history:
                handle.write(json.dumps(row) + "\n")

        csv_path = run_dir / "train_epoch_history.csv"
        headers = ["epoch", "train_loss", "val_balanced_accuracy", "val_roc_auc", "threshold", "lr"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(",".join(headers) + "\n")
            for row in history:
                values = [str(row.get(key, "")) for key in headers]
                handle.write(",".join(values) + "\n")

    runtime_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "epochs_ran": int(train_summary.get("epochs_ran", 0) or 0),
        "best_epoch": int(train_summary.get("best_epoch", -1) or -1),
        "best_val_balanced_accuracy": _safe_float(train_summary.get("best_val_balanced_accuracy")),
        "best_val_auc": _safe_float(train_summary.get("best_val_auc")),
        "best_threshold": _safe_float(train_summary.get("best_threshold")),
        "optimizer": train_summary.get("optimizer", {}),
        "provenance": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "config_path": cfg.get("_meta", {}).get("config_path", ""),
            "dataset_version": cfg.get("data", {}).get("dataset_version", ""),
            "seed": cfg.get("project", {}).get("seed"),
        },
    }
    (run_dir / "train_runtime_summary.json").write_text(
        json.dumps(runtime_summary, indent=2),
        encoding="utf-8",
    )


def _binary_metrics_for_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
    accuracy = (tp + tn) / max(1, len(y_true))
    bal_acc = balanced_accuracy(y_true, y_pred)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], headers: Iterable[str]) -> None:
    _ensure_dir(path)
    header_list = list(headers)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header_list) + "\n")
        for row in rows:
            values: List[str] = []
            for key in header_list:
                value = row.get(key, "")
                if isinstance(value, str):
                    values.append(value.replace(",", " "))
                else:
                    values.append(str(value))
            handle.write(",".join(values) + "\n")


def _calibration_report(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, Any]] = []
    ece = 0.0
    total = max(1, len(y_true))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        count = int(np.sum(mask))
        if count == 0:
            bins.append(
                {
                    "bin_index": i,
                    "prob_min": float(lo),
                    "prob_max": float(hi),
                    "count": 0,
                    "mean_confidence": None,
                    "empirical_accuracy": None,
                    "gap": None,
                }
            )
            continue
        mean_conf = float(np.mean(probs[mask]))
        emp_acc = float(np.mean(y_true[mask]))
        gap = abs(mean_conf - emp_acc)
        ece += (count / total) * gap
        bins.append(
            {
                "bin_index": i,
                "prob_min": float(lo),
                "prob_max": float(hi),
                "count": count,
                "mean_confidence": mean_conf,
                "empirical_accuracy": emp_acc,
                "gap": float(gap),
            }
        )
    return {
        "n_bins": n_bins,
        "ece": float(ece),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "bins": bins,
    }


def write_eval_research_artifacts(
    run_dir: Path,
    metrics: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    model_name: str,
) -> None:
    if not predictions:
        return

    y_true = np.array([int(row["true_label"]) for row in predictions], dtype=int)
    probs = np.array([float(row["prob_fake"]) for row in predictions], dtype=float)
    paths = [str(row.get("path", "")) for row in predictions]
    threshold = float(metrics.get("threshold", 0.5))
    y_pred = (probs >= threshold).astype(int)

    # 1) Threshold sweep
    sweep_thresholds = np.linspace(0.05, 0.95, 19)
    sweep_rows = [_binary_metrics_for_threshold(y_true, probs, float(t)) for t in sweep_thresholds]
    _write_csv(
        run_dir / "threshold_sweep.csv",
        sweep_rows,
        headers=["threshold", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"],
    )
    (run_dir / "threshold_sweep.json").write_text(json.dumps(sweep_rows, indent=2), encoding="utf-8")

    # 2) Confusion + class-wise metrics
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    confusion_payload = {"threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    (run_dir / "confusion_counts.json").write_text(json.dumps(confusion_payload, indent=2), encoding="utf-8")

    # class 1 (fake)
    fake_precision = tp / (tp + fp + 1e-12)
    fake_recall = tp / (tp + fn + 1e-12)
    fake_f1 = (2.0 * fake_precision * fake_recall) / (fake_precision + fake_recall + 1e-12)
    fake_support = int(np.sum(y_true == 1))
    # class 0 (real)
    real_precision = tn / (tn + fn + 1e-12)
    real_recall = tn / (tn + fp + 1e-12)
    real_f1 = (2.0 * real_precision * real_recall) / (real_precision + real_recall + 1e-12)
    real_support = int(np.sum(y_true == 0))

    classwise_payload = {
        "threshold": threshold,
        "real": {
            "precision": float(real_precision),
            "recall": float(real_recall),
            "f1": float(real_f1),
            "support": real_support,
        },
        "fake": {
            "precision": float(fake_precision),
            "recall": float(fake_recall),
            "f1": float(fake_f1),
            "support": fake_support,
        },
    }
    (run_dir / "classwise_metrics.json").write_text(json.dumps(classwise_payload, indent=2), encoding="utf-8")

    # 3) Calibration
    calibration = _calibration_report(y_true=y_true, probs=probs, n_bins=10)
    (run_dir / "calibration_report.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    # 4) Probability summary by true class
    def _summary(arr: np.ndarray) -> Dict[str, float]:
        if len(arr) == 0:
            return {"count": 0, "mean": float("nan"), "std": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan")}
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p25": float(np.quantile(arr, 0.25)),
            "p50": float(np.quantile(arr, 0.50)),
            "p75": float(np.quantile(arr, 0.75)),
        }

    prob_summary = {
        "true_real": _summary(probs[y_true == 0]),
        "true_fake": _summary(probs[y_true == 1]),
    }
    (run_dir / "probability_summary_by_class.json").write_text(json.dumps(prob_summary, indent=2), encoding="utf-8")

    # 5) Error slices
    fp_rows = [
        {"path": paths[i], "true_label": int(y_true[i]), "pred_label": int(y_pred[i]), "prob_fake": float(probs[i])}
        for i in range(len(y_true))
        if y_true[i] == 0 and y_pred[i] == 1
    ]
    fn_rows = [
        {"path": paths[i], "true_label": int(y_true[i]), "pred_label": int(y_pred[i]), "prob_fake": float(probs[i])}
        for i in range(len(y_true))
        if y_true[i] == 1 and y_pred[i] == 0
    ]
    fp_rows = sorted(fp_rows, key=lambda r: r["prob_fake"], reverse=True)
    fn_rows = sorted(fn_rows, key=lambda r: r["prob_fake"])
    _write_csv(
        run_dir / "false_positives_top.csv",
        fp_rows[:200],
        headers=["path", "true_label", "pred_label", "prob_fake"],
    )
    _write_csv(
        run_dir / "false_negatives_top.csv",
        fn_rows[:200],
        headers=["path", "true_label", "pred_label", "prob_fake"],
    )

    # 6) Consolidated run report
    run_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "dataset_version": cfg.get("data", {}).get("dataset_version", ""),
        "run_dir": str(run_dir),
        "metrics": metrics,
        "confusion_counts": confusion_payload,
        "classwise_metrics": classwise_payload,
        "calibration": {
            "ece": calibration["ece"],
            "brier_score": calibration["brier_score"],
            "n_bins": calibration["n_bins"],
        },
        "prediction_count": int(len(predictions)),
        "error_counts": {"false_positive": int(len(fp_rows)), "false_negative": int(len(fn_rows))},
        "provenance": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "config_path": cfg.get("_meta", {}).get("config_path", ""),
        },
    }
    (run_dir / "run_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
