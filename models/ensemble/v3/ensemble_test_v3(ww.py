"""Ensemble model testing script for deepfake detection."""
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from ensemble_config_v3 import CONFIG
from ensemble_data_v3 import build_dataloaders
from ensemble_metrics_v3 import evaluate_predictions
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
    swin_t = float(ckpt.get("swin_threshold", 0.5))
    eff_t = float(ckpt.get("efficientnet_threshold", 0.5))
    base_t = float(ckpt.get("base_threshold", 0.5))
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
    base_probs = 0.5 * swin_probs + 0.5 * eff_probs

    raw_swin = evaluate_predictions(y_bin, swin_probs, 0.5)
    raw_eff = evaluate_predictions(y_bin, eff_probs, 0.5)
    raw_base = evaluate_predictions(y_bin, base_probs, 0.5)

    searched_swin = evaluate_predictions(y_bin, swin_probs, swin_t)
    searched_eff = evaluate_predictions(y_bin, eff_probs, eff_t)
    searched_base = evaluate_predictions(y_bin, base_probs, base_t)

    print("Classes:", classes)
    print("Mapping:", class_to_idx)
    print(f"Thresholds | Swin={swin_t:.2f} EfficientNet={eff_t:.2f} Base50={base_t:.2f}")
    print("Total test time (s):", round(total_test_time, 2))
    print("True counts (real=0,fake=1):", raw_base["true_counts"], "Total:", raw_base["true_counts"].sum())

    print("\nRAW @ 0.50")
    print(
        f"Swin | Acc={raw_swin['acc']:.4f} | Balanced Acc={raw_swin['balanced_acc']:.4f}"
        + (f" | AUC={raw_swin['auc']:.4f}" if raw_swin["auc"] is not None else "")
    )
    print(
        f"EfficientNet | Acc={raw_eff['acc']:.4f} | Balanced Acc={raw_eff['balanced_acc']:.4f}"
        + (f" | AUC={raw_eff['auc']:.4f}" if raw_eff["auc"] is not None else "")
    )
    print(
        f"Base 50/50 | Acc={raw_base['acc']:.4f} | Balanced Acc={raw_base['balanced_acc']:.4f}"
        + (f" | AUC={raw_base['auc']:.4f}" if raw_base["auc"] is not None else "")
    )
    print("Pred counts @0.5 (base50):", raw_base["pred_counts"])

    print("\nSEARCHED THRESHOLDS")
    print(
        f"Swin | thr={swin_t:.2f} | Acc={searched_swin['acc']:.4f} | Balanced Acc={searched_swin['balanced_acc']:.4f}"
        + (f" | AUC={searched_swin['auc']:.4f}" if searched_swin["auc"] is not None else "")
    )
    print(
        f"EfficientNet | thr={eff_t:.2f} | Acc={searched_eff['acc']:.4f} | Balanced Acc={searched_eff['balanced_acc']:.4f}"
        + (f" | AUC={searched_eff['auc']:.4f}" if searched_eff["auc"] is not None else "")
    )
    print(
        f"Base 50/50 | thr={base_t:.2f} | Acc={searched_base['acc']:.4f} | Balanced Acc={searched_base['balanced_acc']:.4f}"
        + (f" | AUC={searched_base['auc']:.4f}" if searched_base["auc"] is not None else "")
    )
    print("Pred counts @searched (base50):", searched_base["pred_counts"])

    print("\nBase 50/50 confusion matrix @ searched threshold")
    print(searched_base["cm"])
    print("Classification Report:")
    print(searched_base["report"])


if __name__ == "__main__":
    main()
