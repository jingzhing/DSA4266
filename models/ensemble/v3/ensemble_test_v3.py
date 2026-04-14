"""Ensemble model testing script with per-model comparison and average prediction analysis."""
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


def summarize_probs(name, probs, y_bin):
    real_mask = y_bin == 0
    fake_mask = y_bin == 1

    real_mean = float(np.mean(probs[real_mask])) if np.any(real_mask) else float("nan")
    fake_mean = float(np.mean(probs[fake_mask])) if np.any(fake_mask) else float("nan")
    overall_mean = float(np.mean(probs)) if len(probs) else float("nan")

    print(f"{name} average predicted fake-probability:")
    print(f"  real samples   : {real_mean:.4f}")
    print(f"  fake samples   : {fake_mean:.4f}")
    print(f"  overall average: {overall_mean:.4f}")


def print_model_block(title, metrics_05, metrics_search, threshold, probs, y_bin):
    print(f"\n{title}")
    summarize_probs(title, probs, y_bin)

    print("  RAW @ 0.50")
    msg = (
        f"    Acc={metrics_05['acc']:.4f} | "
        f"Balanced Acc={metrics_05['balanced_acc']:.4f}"
    )
    if metrics_05["auc"] is not None:
        msg += f" | AUC={metrics_05['auc']:.4f}"
    print(msg)
    print(f"    Pred counts @0.50: {metrics_05['pred_counts']}")

    print(f"  SEARCHED @ {threshold:.2f}")
    msg = (
        f"    Acc={metrics_search['acc']:.4f} | "
        f"Balanced Acc={metrics_search['balanced_acc']:.4f}"
    )
    if metrics_search["auc"] is not None:
        msg += f" | AUC={metrics_search['auc']:.4f}"
    print(msg)
    print(f"    Pred counts @searched: {metrics_search['pred_counts']}")


def main():
    cfg = CONFIG
    init_run_logging(cfg["log_dir"], cfg["test_log_name"])
    log_section("ENSEMBLE V3 TESTING WITH MODEL COMPARISON")

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
    def safe_threshold(value, default=0.5):
        return default if value is None else float(value)

    swin_t = safe_threshold(ckpt.get("swin_threshold"))
    eff_t = safe_threshold(ckpt.get("efficientnet_threshold"))
    base_t = safe_threshold(ckpt.get("base_threshold"))
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

    print_model_block("SWIN ONLY", raw_swin, searched_swin, swin_t, swin_probs, y_bin)
    print_model_block("EFFICIENTNET ONLY", raw_eff, searched_eff, eff_t, eff_probs, y_bin)
    print_model_block("COMBINED 50/50", raw_base, searched_base, base_t, base_probs, y_bin)

    print("\nCOMBINED 50/50 confusion matrix @ searched threshold")
    print(searched_base["cm"])
    print("Classification Report:")
    print(searched_base["report"])


if __name__ == "__main__":
    main()
