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


def find_best_threshold(y_true, probs, metric="balanced_acc"):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_s = -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        s = balanced_accuracy(y_true, pred)
        if s > best_s:
            best_s = s
            best_t = float(t)
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
