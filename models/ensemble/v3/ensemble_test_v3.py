import os
import time
import numpy as np
import torch
from tqdm import tqdm

from ensemble_config_v3 import CONFIG
from ensemble_data_v3 import build_dataloaders
from ensemble_metrics_v3 import evaluate_predictions, combine_probabilities
from ensemble_model_utils_v3 import build_ensemble_model
from logging_utils_v3 import init_run_logging, log_section


def move_inputs_to_device(inputs, device):
    return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}


def main():
    cfg = CONFIG
    init_run_logging(cfg["log_dir"], cfg["test_log_name"])
    log_section("ENSEMBLE V3 TESTING")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "best_ensemble.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}. Train first.")

    print("Device:", device)
    print("Loading checkpoint:", ckpt_path)

    bundle = build_dataloaders(cfg)
    test_loader = bundle["test_loader"]
    test_ds = bundle["test_ds"]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_t = float(ckpt.get("best_threshold", cfg["default_threshold"]))
    best_weights = ckpt.get("best_ensemble_weights", cfg["ensemble"]["initial_weights"])
    classes = ckpt.get("classes", test_ds.classes)
    class_to_idx = ckpt.get("class_to_idx", test_ds.class_to_idx)

    if class_to_idx != test_ds.class_to_idx:
        raise RuntimeError(f"Class mapping mismatch. ckpt={class_to_idx} test={test_ds.class_to_idx}")
    if "fake" not in class_to_idx or "real" not in class_to_idx:
        raise RuntimeError(f"Expected classes fake/real. Got: {classes} mapping={class_to_idx}")

    fake_idx = class_to_idx["fake"]
    model = build_ensemble_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_y = []
    all_swin_probs = []
    all_eff_probs = []

    test_start = time.time()
    with torch.no_grad():
        for step, (batch_inputs, y) in enumerate(tqdm(test_loader, desc="Testing"), start=1):
            batch_inputs = move_inputs_to_device(batch_inputs, device)
            out = model(batch_inputs)
            all_swin_probs.append(out["swin_probs"].squeeze(1).cpu().numpy())
            all_eff_probs.append(out["efficientnet_probs"].squeeze(1).cpu().numpy())
            all_y.append(y.numpy())
            if step % max(1, len(test_loader) // 4) == 0:
                print(f"Processed {step}/{len(test_loader)} test batches")

    total_test_time = time.time() - test_start

    y_orig = np.concatenate(all_y).astype(int)
    y_bin = (y_orig == fake_idx).astype(int)
    swin_probs = np.concatenate(all_swin_probs)
    eff_probs = np.concatenate(all_eff_probs)
    ensemble_probs = combine_probabilities(swin_probs, eff_probs, float(best_weights["swin"]))

    ensemble_eval = evaluate_predictions(y_bin, ensemble_probs, best_t)
    swin_eval = evaluate_predictions(y_bin, swin_probs, best_t)
    eff_eval = evaluate_predictions(y_bin, eff_probs, best_t)

    print("Classes:", classes)
    print("Mapping:", class_to_idx)
    print("Using learned ensemble weights:", best_weights)
    print("Using threshold:", best_t)
    print("Total test time (s):", round(total_test_time, 2))
    print("True counts (real=0,fake=1):", ensemble_eval["true_counts"], "Total:", ensemble_eval["true_counts"].sum())
    print("Pred counts (real=0,fake=1):", ensemble_eval["pred_counts"], "Total:", ensemble_eval["pred_counts"].sum())
    print(f"Accuracy (on y_bin): {ensemble_eval['acc']:.4f}")
    print(f"Balanced Acc: {ensemble_eval['balanced_acc']:.4f}")
    if ensemble_eval["auc"] is not None:
        print(f"AUC (fake=1): {ensemble_eval['auc']:.4f}")
    print("Confusion Matrix (rows true [real,fake], cols pred [real,fake]):")
    print(ensemble_eval["cm"])
    print("Classification Report:")
    print(ensemble_eval["report"])

    print("\nComponent Model Comparison")
    print(
        f"Swin Acc: {swin_eval['acc']:.4f} | Swin Balanced Acc: {swin_eval['balanced_acc']:.4f}" +
        (f" | Swin AUC: {swin_eval['auc']:.4f}" if swin_eval["auc"] is not None else "")
    )
    print(
        f"EfficientNet Acc: {eff_eval['acc']:.4f} | EfficientNet Balanced Acc: {eff_eval['balanced_acc']:.4f}" +
        (f" | EfficientNet AUC: {eff_eval['auc']:.4f}" if eff_eval["auc"] is not None else "")
    )
    print(
        f"Learned Ensemble Acc: {ensemble_eval['acc']:.4f} | Learned Ensemble Balanced Acc: {ensemble_eval['balanced_acc']:.4f}" +
        (f" | Learned Ensemble AUC: {ensemble_eval['auc']:.4f}" if ensemble_eval["auc"] is not None else "")
    )


if __name__ == "__main__":
    main()
