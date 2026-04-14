import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


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


def _metric_score(metric_name, y_true, pred, probs):
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


def find_best_threshold(
    y_true,
    probs,
    metric="balanced_acc",
    threshold_start=0.01,
    threshold_end=0.99,
    threshold_step=0.01,
):
    thresholds = np.arange(threshold_start, threshold_end + 1e-12, threshold_step)
    best_t = 0.5
    best_s = -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        s = _metric_score(metric, y_true, pred, probs)
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


def combine_probabilities(swin_probs, eff_probs, swin_weight):
    swin_weight = float(swin_weight)
    eff_weight = 1.0 - swin_weight
    probs = swin_weight * swin_probs + eff_weight * eff_probs
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def search_best_ensemble_weight(
    y_true,
    swin_probs,
    eff_probs,
    metric="balanced_acc",
    weight_start=0.0,
    weight_end=1.0,
    weight_step=0.02,
    threshold_start=0.01,
    threshold_end=0.99,
    threshold_step=0.01,
):
    candidates = []
    weight_values = np.arange(weight_start, weight_end + 1e-12, weight_step)

    default_probs = combine_probabilities(swin_probs, eff_probs, 0.5)
    default_t, default_s = find_best_threshold(
        y_true,
        default_probs,
        metric=metric,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
    )

    best = {
        "swin_weight": 0.5,
        "efficientnet_weight": 0.5,
        "threshold": default_t,
        "score": default_s,
        "auc": None,
        "balanced_acc": None,
        "acc": None,
        "probs": default_probs,
    }

    for swin_weight in weight_values:
        probs = combine_probabilities(swin_probs, eff_probs, swin_weight)
        threshold, score = find_best_threshold(
            y_true,
            probs,
            metric=metric,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            threshold_step=threshold_step,
        )
        metrics = evaluate_predictions(y_true, probs, threshold)
        row = {
            "swin_weight": float(round(swin_weight, 6)),
            "efficientnet_weight": float(round(1.0 - swin_weight, 6)),
            "threshold": float(threshold),
            "score": float(score),
            "auc": None if metrics["auc"] is None else float(metrics["auc"]),
            "balanced_acc": float(metrics["balanced_acc"]),
            "acc": float(metrics["acc"]),
            "probs": probs,
        }
        candidates.append(row)
        if row["score"] > best["score"]:
            best = row

    ranked = sorted(
        candidates,
        key=lambda x: (
            x["score"],
            x["auc"] if x["auc"] is not None else -1.0,
            x["acc"],
        ),
        reverse=True,
    )
    return best, ranked
