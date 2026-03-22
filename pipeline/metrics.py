from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    return float(0.5 * (tpr + tnr))


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_score = -1.0
    for threshold in thresholds:
        pred = (probs >= threshold).astype(int)
        score = balanced_accuracy(y_true, pred)
        if score > best_score:
            best_score = float(score)
            best_t = float(threshold)
    return best_t, best_score


def compute_binary_metrics(
    y_true: np.ndarray,
    probs_fake: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    pred = (probs_fake >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": balanced_accuracy(y_true, pred),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs_fake))
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def compute_confusion(y_true: np.ndarray, probs_fake: np.ndarray, threshold: float) -> np.ndarray:
    pred = (probs_fake >= threshold).astype(int)
    return confusion_matrix(y_true, pred, labels=[0, 1])


def save_confusion_matrix_png(path: str | Path, cm: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["real(0)", "fake(1)"])
    ax.set_yticklabels(["real(0)", "fake(1)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def metrics_to_rows(metrics: Dict[str, float]) -> Iterable[Dict[str, str]]:
    rows = []
    for key, value in metrics.items():
        rows.append({"metric": key, "value": value})
    return rows
