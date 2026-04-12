import json
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from ensemble_config_v2 import CONFIG
from ensemble_data_v2 import build_dataloaders, set_seed
from ensemble_metrics_v2 import evaluate_predictions, find_best_threshold
from ensemble_model_utils_v2 import build_ensemble_model, build_optimizer
from logging_utils_v2 import init_run_logging, log_section


def smooth_binary_labels(y, smoothing=0.0):
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing


def move_inputs_to_device(inputs, device):
    return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}


def eval_on_loader(model, loader, device, fake_idx, threshold_metric):
    model.eval()
    all_y = []
    all_ensemble_probs = []
    all_swin_probs = []
    all_eff_probs = []

    with torch.no_grad():
        for batch_inputs, y in tqdm(loader, desc="Validation", leave=False):
            batch_inputs = move_inputs_to_device(batch_inputs, device)
            out = model(batch_inputs)
            all_ensemble_probs.append(out["ensemble_probs"].squeeze(1).cpu().numpy())
            all_swin_probs.append(out["swin_probs"].squeeze(1).cpu().numpy())
            all_eff_probs.append(out["efficientnet_probs"].squeeze(1).cpu().numpy())
            all_y.append(y.numpy())

    y_orig = np.concatenate(all_y).astype(int)
    y_bin = (y_orig == fake_idx).astype(int)
    ensemble_probs = np.concatenate(all_ensemble_probs)
    swin_probs = np.concatenate(all_swin_probs)
    eff_probs = np.concatenate(all_eff_probs)

    best_t, best_score = find_best_threshold(y_bin, ensemble_probs, metric=threshold_metric)
    return {
        "best_threshold": best_t,
        "best_score": best_score,
        "y_bin": y_bin,
        "ensemble_probs": ensemble_probs,
        "swin_probs": swin_probs,
        "efficientnet_probs": eff_probs,
        "ensemble_eval": evaluate_predictions(y_bin, ensemble_probs, best_t),
        "swin_eval": evaluate_predictions(y_bin, swin_probs, best_t),
        "efficientnet_eval": evaluate_predictions(y_bin, eff_probs, best_t),
    }


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    init_run_logging(cfg["log_dir"], cfg["train_log_name"])
    log_section("ENSEMBLE V2 TRAINING")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("Device:", device)
    print("Ensemble method:", cfg["ensemble_method"])
    print("Ensemble weights:", cfg["ensemble_weights"])
    print("Swin config:", cfg["swin"])
    print("EfficientNet config:", cfg["efficientnet"])

    bundle = build_dataloaders(cfg)
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    classes = bundle["classes"]
    class_to_idx = bundle["class_to_idx"]
    fake_idx = class_to_idx["fake"]

    log_section("DATA SUMMARY")
    print("Classes:", classes)
    print("Mapping:", class_to_idx)
    print("Total train folder size:", len(bundle["full_ds"]))
    print("Train split size:", len(bundle["train_ds"]))
    print("Val split size:", len(bundle["val_ds"]))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    train_targets_orig = np.array(bundle["full_ds"].targets)[bundle["base_train_ds"].indices]
    y_train_bin = (train_targets_orig == fake_idx).astype(int)
    fake_count = int(y_train_bin.sum())
    real_count = int(len(y_train_bin) - fake_count)
    print("Train split counts (real=0,fake=1):", [real_count, fake_count])

    model = build_ensemble_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("Using BCEWithLogitsLoss on ensemble logits")
    print("Swin label smoothing:", cfg["swin"].get("label_smoothing", 0.0))
    print("EfficientNet label smoothing kept at:", cfg["efficientnet"].get("label_smoothing", 0.0))

    best_val_score = -1.0
    best_path = os.path.join(cfg["checkpoint_dir"], "best_ensemble.pt")
    train_history = []

    for epoch in range(cfg["epochs"]):
        log_section(f"EPOCH {epoch + 1}/{cfg['epochs']}")
        model.train()
        losses = []
        epoch_start = time.time()

        for step, (batch_inputs, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"), start=1):
            batch_inputs = move_inputs_to_device(batch_inputs, device)
            y_bin = (y == fake_idx).float().to(device, non_blocking=True).unsqueeze(1)

            eff_targets = smooth_binary_labels(y_bin, cfg["efficientnet"].get("label_smoothing", 0.0))
            swin_targets = smooth_binary_labels(y_bin, cfg["swin"].get("label_smoothing", 0.0))
            ensemble_targets = y_bin

            optimizer.zero_grad(set_to_none=True)
            out = model(batch_inputs)

            ensemble_loss = loss_fn(out["ensemble_logits"], ensemble_targets)
            swin_loss = loss_fn(out["swin_logits"], swin_targets)
            eff_loss = loss_fn(out["efficientnet_logits"], eff_targets)
            loss = 0.6 * ensemble_loss + 0.2 * swin_loss + 0.2 * eff_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step % max(1, len(train_loader) // 5) == 0:
                print(
                    f"step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} ensemble_loss={ensemble_loss.item():.4f} "
                    f"swin_loss={swin_loss.item():.4f} eff_loss={eff_loss.item():.4f}"
                )

        val_out = eval_on_loader(model, val_loader, device, fake_idx, cfg["threshold_metric"])
        ensemble_eval = val_out["ensemble_eval"]
        swin_eval = val_out["swin_eval"]
        eff_eval = val_out["efficientnet_eval"]
        epoch_time = time.time() - epoch_start
        avg_loss = float(np.mean(losses)) if losses else float("nan")

        print(
            f"epoch={epoch + 1} train_loss={avg_loss:.4f} epoch_time={epoch_time:.1f}s "
            f"val_bal_acc={ensemble_eval['balanced_acc']:.4f} best_t={val_out['best_threshold']:.2f}"
            + (f" val_auc={ensemble_eval['auc']:.4f}" if ensemble_eval["auc"] is not None else "")
        )
        print("Val true counts (real=0,fake=1):", ensemble_eval["true_counts"], "Val pred counts:", ensemble_eval["pred_counts"])
        print(
            f"Val component scores | Ensemble bal_acc={ensemble_eval['balanced_acc']:.4f} "
            f"Swin bal_acc={swin_eval['balanced_acc']:.4f} "
            f"EfficientNet bal_acc={eff_eval['balanced_acc']:.4f}"
        )

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "epoch_time_sec": epoch_time,
            "ensemble_bal_acc": float(ensemble_eval["balanced_acc"]),
            "swin_bal_acc": float(swin_eval["balanced_acc"]),
            "efficientnet_bal_acc": float(eff_eval["balanced_acc"]),
            "best_threshold": float(val_out["best_threshold"]),
            "ensemble_auc": None if ensemble_eval["auc"] is None else float(ensemble_eval["auc"]),
        })

        if ensemble_eval["balanced_acc"] > best_val_score:
            best_val_score = ensemble_eval["balanced_acc"]
            payload = {
                "model": model.state_dict(),
                "cfg": cfg,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "best_threshold": float(val_out["best_threshold"]),
                "val_score": float(ensemble_eval["balanced_acc"]),
                "val_auc": None if ensemble_eval["auc"] is None else float(ensemble_eval["auc"]),
                "swin_val_auc": None if swin_eval["auc"] is None else float(swin_eval["auc"]),
                "efficientnet_val_auc": None if eff_eval["auc"] is None else float(eff_eval["auc"]),
            }
            if cfg.get("save_val_outputs", True):
                payload["val_outputs"] = {
                    "y_bin": val_out["y_bin"].tolist(),
                    "ensemble_probs": val_out["ensemble_probs"].tolist(),
                    "swin_probs": val_out["swin_probs"].tolist(),
                    "efficientnet_probs": val_out["efficientnet_probs"].tolist(),
                }
            torch.save(payload, best_path)
            print("Saved best checkpoint:", best_path)

    summary_path = os.path.join(cfg["output_dir"], "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_score": float(best_val_score),
                "checkpoint": best_path,
                "ensemble_method": cfg["ensemble_method"],
                "ensemble_weights": cfg["ensemble_weights"],
                "swin_config": cfg["swin"],
                "efficientnet_config": cfg["efficientnet"],
                "epochs": train_history,
            },
            f,
            indent=2,
        )
    print("Saved training summary:", summary_path)


if __name__ == "__main__":
    main()
