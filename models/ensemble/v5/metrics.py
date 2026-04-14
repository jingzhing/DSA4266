import csv
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def balanced_accuracy(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    return 0.5 * (tpr + tnr)


def metric_score(metric_name, y_true, pred, probs):
    if metric_name == "balanced_acc":
        return balanced_accuracy(y_true, pred)
    if metric_name == "acc":
        return accuracy_score(y_true, pred)
    if metric_name == "auc":
        try:
            return roc_auc_score(y_true, probs)
        except Exception:
            return -1.0
    raise ValueError(f"Unsupported metric: {metric_name}")


def find_best_threshold(y_true, probs, metric="balanced_acc", threshold_start=0.1, threshold_end=0.9, threshold_step=0.02):
    thresholds = np.arange(threshold_start, threshold_end + 1e-12, threshold_step)
    best_t = 0.5
    best_s = -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        s = metric_score(metric, y_true, pred, probs)
        if s > best_s:
            best_s = float(s)
            best_t = float(round(t, 6))
    return best_t, best_s


def evaluate_predictions(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = None
    return {
        "acc": accuracy_score(y_true, pred),
        "balanced_acc": balanced_accuracy(y_true, pred),
        "auc": auc,
        "true_counts": np.bincount(y_true, minlength=2),
        "pred_counts": np.bincount(pred, minlength=2),
        "cm": confusion_matrix(y_true, pred),
        "report": classification_report(
            y_true,
            pred,
            target_names=["real(0)", "fake(1)"],
            digits=4,
            zero_division=0,
        ),
        "pred": pred,
    }


def summarize_probabilities(probs, y_true):
    real_mask = y_true == 0
    fake_mask = y_true == 1

    def _stats(arr):
        if arr.size == 0:
            return {"mean": None, "std": None, "q05": None, "q25": None, "q50": None, "q75": None, "q95": None}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "q05": float(np.quantile(arr, 0.05)),
            "q25": float(np.quantile(arr, 0.25)),
            "q50": float(np.quantile(arr, 0.50)),
            "q75": float(np.quantile(arr, 0.75)),
            "q95": float(np.quantile(arr, 0.95)),
        }

    return {
        "real": _stats(probs[real_mask]),
        "fake": _stats(probs[fake_mask]),
        "overall": _stats(probs),
        "mean_gap_fake_minus_real": None if not np.any(real_mask) or not np.any(fake_mask) else float(np.mean(probs[fake_mask]) - np.mean(probs[real_mask])),
    }


def make_mistake_rows(paths, y_true, probs, pred, top_k=50):
    rows = []
    for path, yt, pr, pd in zip(paths, y_true, probs, pred):
        if int(yt) == int(pd):
            continue
        confidence = pr if pd == 1 else 1.0 - pr
        rows.append({
            "path": path,
            "true_label": int(yt),
            "pred_label": int(pd),
            "prob_fake": float(pr),
            "confidence": float(confidence),
            "error_type": "fp" if yt == 0 and pd == 1 else "fn",
        })
    rows.sort(key=lambda x: x["confidence"], reverse=True)
    return rows[:top_k]


def save_rows_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def combine_logits(logits_a, logits_b, weight_a):
    weight_a = float(weight_a)
    weight_b = 1.0 - weight_a
    return weight_a * logits_a + weight_b * logits_b


def combine_probabilities(probs_a, probs_b, weight_a):
    weight_a = float(weight_a)
    weight_b = 1.0 - weight_a
    probs = weight_a * probs_a + weight_b * probs_b
    return np.clip(probs, 1e-6, 1 - 1e-6)


def search_best_weight(y_true, logits_a, logits_b, probs_a, probs_b, search_on="logits", metric="balanced_acc",
                       weight_start=0.0, weight_end=1.0, weight_step=0.05,
                       threshold_start=0.1, threshold_end=0.9, threshold_step=0.02):
    weight_values = np.arange(weight_start, weight_end + 1e-12, weight_step)
    ranked = []
    best = None

    for w in weight_values:
        if search_on == "logits":
            fused_logits = combine_logits(logits_a, logits_b, w)
            fused_probs = sigmoid(fused_logits)
        elif search_on == "probs":
            fused_probs = combine_probabilities(probs_a, probs_b, w)
        else:
            raise ValueError(f"Unsupported search_on: {search_on}")

        threshold, score = find_best_threshold(
            y_true, fused_probs, metric=metric,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            threshold_step=threshold_step,
        )
        metrics = evaluate_predictions(y_true, fused_probs, threshold)
        row = {
            "weight_a": float(w),
            "weight_b": float(1.0 - w),
            "threshold": float(threshold),
            "score": float(score),
            "acc": float(metrics["acc"]),
            "balanced_acc": float(metrics["balanced_acc"]),
            "auc": None if metrics["auc"] is None else float(metrics["auc"]),
        }
        ranked.append(row)
        if best is None or row["score"] > best["score"]:
            best = row

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return best, ranked


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
