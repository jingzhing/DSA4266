import json
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from ensemble_config_v3 import CONFIG
from ensemble_data_v3 import build_dataloaders, set_seed
from ensemble_metrics_v3 import evaluate_predictions, find_best_threshold
from ensemble_model_utils_v3 import build_ensemble_model, build_optimizer
from logging_utils_v3 import init_run_logging, log_section


def smooth_binary_labels(y, smoothing=0.0):
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing


def move_inputs_to_device(inputs, device):
    return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}


def threshold_search_metrics(y_true, probs, cfg):
    threshold, _ = find_best_threshold(
        y_true,
        probs,
        metric=cfg.get("threshold_metric", "balanced_acc"),
        threshold_start=cfg.get("threshold_search", {}).get("start", 0.30),
        threshold_end=cfg.get("threshold_search", {}).get("end", 0.70),
        threshold_step=cfg.get("threshold_search", {}).get("step", 0.01),
    )
    metrics = evaluate_predictions(y_true, probs, threshold)
    return threshold, metrics


def eval_on_loader(model, loader, device, fake_idx, cfg):
    model.eval()
    all_y = []
    all_swin_probs = []
    all_eff_probs = []

    with torch.no_grad():
        for batch_inputs, y in tqdm(loader, desc="Validation", leave=False):
            batch_inputs = move_inputs_to_device(batch_inputs, device)
            out = model(batch_inputs)
            all_swin_probs.append(out["swin_probs"].squeeze(1).cpu().numpy())
            all_eff_probs.append(out["efficientnet_probs"].squeeze(1).cpu().numpy())
            all_y.append(y.numpy())

    y_orig = np.concatenate(all_y).astype(int)
    y_bin = (y_orig == fake_idx).astype(int)
    swin_probs = np.concatenate(all_swin_probs)
    eff_probs = np.concatenate(all_eff_probs)
    base_probs = 0.5 * swin_probs + 0.5 * eff_probs

    raw = {
        "swin": evaluate_predictions(y_bin, swin_probs, 0.5),
        "efficientnet": evaluate_predictions(y_bin, eff_probs, 0.5),
        "base_ensemble": evaluate_predictions(y_bin, base_probs, 0.5),
    }

    searched = {}
    searched["swin_threshold"], searched["swin_eval"] = threshold_search_metrics(y_bin, swin_probs, cfg)
    searched["efficientnet_threshold"], searched["efficientnet_eval"] = threshold_search_metrics(y_bin, eff_probs, cfg)
    searched["base_threshold"], searched["base_ensemble_eval"] = threshold_search_metrics(y_bin, base_probs, cfg)

    return {
        "y_bin": y_bin,
        "swin_probs": swin_probs,
        "efficientnet_probs": eff_probs,
        "base_ensemble_probs": base_probs,
        "raw": raw,
        **searched,
    }


def print_label_sanity(train_loader, class_to_idx, fake_idx, num_batches=3):
    print("LABEL SANITY CHECK")
    print("class_to_idx:", class_to_idx)
    print("fake_idx:", fake_idx)

    for i, (_, y) in enumerate(train_loader):
        raw_unique, raw_counts = torch.unique(y, return_counts=True)
        y_bin = (y == fake_idx).int()
        bin_unique, bin_counts = torch.unique(y_bin, return_counts=True)
        print(f"batch={i+1} raw_unique={raw_unique.tolist()} raw_counts={raw_counts.tolist()}")
        print(f"batch={i+1} bin_unique={bin_unique.tolist()} bin_counts={bin_counts.tolist()}")
        print(f"batch={i+1} first_10_raw={y[:10].tolist()}")
        print(f"batch={i+1} first_10_bin={y_bin[:10].tolist()}")
        if i + 1 >= num_batches:
            break


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    init_run_logging(cfg["log_dir"], cfg["train_log_name"])
    log_section("ENSEMBLE V3 TRAINING")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("Device:", device)
    print("Base ensemble method:", cfg["ensemble"]["base_method"])
    print("Initial ensemble weights:", cfg["ensemble"]["initial_weights"])
    print("Post-training weight search enabled:", cfg["ensemble"].get("learn_weights_post_training", False))
    print("Swin config:", cfg["swin"])
    print("EfficientNet config:", cfg["efficientnet"])
    print("Threshold search config:", cfg.get("threshold_search", {}))
    print("Validation split path:", cfg.get("val_split_path"))

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

    val_targets_orig = np.array(bundle["full_ds"].targets)[bundle["base_val_ds"].indices]
    y_val_bin = (val_targets_orig == fake_idx).astype(int)
    val_fake_count = int(y_val_bin.sum())
    val_real_count = int(len(y_val_bin) - val_fake_count)
    print("Val split counts (real=0,fake=1):", [val_real_count, val_fake_count])

    print_label_sanity(
        train_loader,
        class_to_idx=class_to_idx,
        fake_idx=fake_idx,
        num_batches=cfg.get("debug", {}).get("label_sanity_batches", 3),
    )

    model = build_ensemble_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)

    eff_sched_cfg = cfg["efficientnet"].get("scheduler", {})
    scheduler = None
    if eff_sched_cfg.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=eff_sched_cfg.get("mode", "max"),
            factor=eff_sched_cfg.get("factor", 0.5),
            patience=eff_sched_cfg.get("patience", 1),
            min_lr=eff_sched_cfg.get("min_lr", 1e-6),
        )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("Using BCEWithLogitsLoss on branch logits only")
    print("No pos_weight is being used")
    print("Checkpoint selection metric: validation AUC on fixed 50/50 ensemble")

    best_val_auc = -1.0
    best_path = os.path.join(cfg["checkpoint_dir"], "best_ensemble.pt")
    train_history = []

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", False)
    patience = int(early_cfg.get("patience", 2))
    min_delta = float(early_cfg.get("min_delta", 5e-4))
    epochs_without_improve = 0

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

            optimizer.zero_grad(set_to_none=True)
            out = model(batch_inputs)

            swin_loss = loss_fn(out["swin_logits"], swin_targets)
            eff_loss = loss_fn(out["efficientnet_logits"], eff_targets)
            loss = 0.5 * swin_loss + 0.5 * eff_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step % max(1, len(train_loader) // 5) == 0:
                print(
                    f"step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} "
                    f"swin_loss={swin_loss.item():.4f} "
                    f"eff_loss={eff_loss.item():.4f}"
                )

        val_out = eval_on_loader(model, val_loader, device, fake_idx, cfg)
        raw = val_out["raw"]
        base_raw = raw["base_ensemble"]
        base_searched = val_out["base_ensemble_eval"]
        epoch_time = time.time() - epoch_start
        avg_loss = float(np.mean(losses)) if losses else float("nan")

        print(
            f"epoch={epoch + 1} train_loss={avg_loss:.4f} epoch_time={epoch_time:.1f}s "
            f"base_val_auc={base_raw['auc']:.4f} "
            f"base_val_bal_acc_at_0.5={base_raw['balanced_acc']:.4f} "
            f"base_val_bal_acc_searched={base_searched['balanced_acc']:.4f} "
            f"base_best_t={val_out['base_threshold']:.2f}"
        )
        print("Val true counts (real=0,fake=1):", base_raw["true_counts"], "Val pred counts @0.5:", base_raw["pred_counts"])
        print("Val pred counts @searched:", base_searched["pred_counts"])
        print(
            f"Val raw@0.5 | Swin bal_acc={raw['swin']['balanced_acc']:.4f} "
            f"Eff bal_acc={raw['efficientnet']['balanced_acc']:.4f} "
            f"Base50 bal_acc={raw['base_ensemble']['balanced_acc']:.4f}"
        )
        print(
            f"Val searched | Swin thr={val_out['swin_threshold']:.2f} bal_acc={val_out['swin_eval']['balanced_acc']:.4f} "
            f"Eff thr={val_out['efficientnet_threshold']:.2f} bal_acc={val_out['efficientnet_eval']['balanced_acc']:.4f} "
            f"Base50 thr={val_out['base_threshold']:.2f} bal_acc={val_out['base_ensemble_eval']['balanced_acc']:.4f}"
        )

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "epoch_time_sec": epoch_time,
            "base_auc": None if base_raw["auc"] is None else float(base_raw["auc"]),
            "base_bal_acc_at_0_5": float(base_raw["balanced_acc"]),
            "base_bal_acc_searched": float(base_searched["balanced_acc"]),
            "base_best_threshold": float(val_out["base_threshold"]),
            "swin_bal_acc_at_0_5": float(raw["swin"]["balanced_acc"]),
            "efficientnet_bal_acc_at_0_5": float(raw["efficientnet"]["balanced_acc"]),
        })

        current_auc = -1.0 if base_raw["auc"] is None else float(base_raw["auc"])
        if scheduler is not None:
            scheduler.step(current_auc)
            print("Current learning rates:", [group["lr"] for group in optimizer.param_groups])

        if current_auc > best_val_auc + min_delta:
            best_val_auc = current_auc
            epochs_without_improve = 0

            payload = {
                "model": model.state_dict(),
                "cfg": cfg,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "base_ensemble_method": cfg["ensemble"]["base_method"],
                "selection_metric": "base_val_auc",
                "best_threshold": None,
                "best_ensemble_weights": {
                    "swin": 0.5,
                    "efficientnet": 0.5,
                },
                "base_threshold": None,
                "swin_threshold": None,
                "efficientnet_threshold": None,
                "val_score": float(current_auc),
                "val_auc": float(current_auc),
                "raw_val_metrics": {
                    "swin": {
                        "auc": raw["swin"]["auc"],
                        "balanced_acc_at_0_5": raw["swin"]["balanced_acc"],
                    },
                    "efficientnet": {
                        "auc": raw["efficientnet"]["auc"],
                        "balanced_acc_at_0_5": raw["efficientnet"]["balanced_acc"],
                    },
                    "base_ensemble": {
                        "auc": raw["base_ensemble"]["auc"],
                        "balanced_acc_at_0_5": raw["base_ensemble"]["balanced_acc"],
                    },
                },
            }

            if cfg.get("save_val_outputs", True):
                payload["val_outputs"] = {
                    "y_bin": val_out["y_bin"].tolist(),
                    "swin_probs": val_out["swin_probs"].tolist(),
                    "efficientnet_probs": val_out["efficientnet_probs"].tolist(),
                    "base_ensemble_probs": val_out["base_ensemble_probs"].tolist(),
                }

            torch.save(payload, best_path)
            print("Saved best checkpoint by val AUC:", best_path)

        else:
            epochs_without_improve += 1
            print(f"No significant improvement for {epochs_without_improve} epoch(s)")

        if early_enabled and epochs_without_improve >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    if os.path.exists(best_path):
        print("\nPOST-TRAINING THRESHOLD SEARCH ON BEST CHECKPOINT")
        ckpt = torch.load(best_path, map_location="cpu")
        val_outputs = ckpt.get("val_outputs")
        if val_outputs is None:
            raise RuntimeError("No validation outputs were saved in the best checkpoint")

        y_bin = np.array(val_outputs["y_bin"], dtype=int)
        swin_probs = np.array(val_outputs["swin_probs"], dtype=float)
        eff_probs = np.array(val_outputs["efficientnet_probs"], dtype=float)
        base_probs = np.array(val_outputs["base_ensemble_probs"], dtype=float)

        swin_threshold, swin_eval = threshold_search_metrics(y_bin, swin_probs, cfg)
        eff_threshold, eff_eval = threshold_search_metrics(y_bin, eff_probs, cfg)
        base_threshold, base_eval = threshold_search_metrics(y_bin, base_probs, cfg)

        ckpt["swin_threshold"] = float(swin_threshold)
        ckpt["efficientnet_threshold"] = float(eff_threshold)
        ckpt["base_threshold"] = float(base_threshold)
        ckpt["best_threshold"] = float(base_threshold)
        ckpt["post_training_threshold_metrics"] = {
            "swin": {
                "threshold": float(swin_threshold),
                "balanced_acc": float(swin_eval["balanced_acc"]),
                "auc": None if swin_eval["auc"] is None else float(swin_eval["auc"]),
            },
            "efficientnet": {
                "threshold": float(eff_threshold),
                "balanced_acc": float(eff_eval["balanced_acc"]),
                "auc": None if eff_eval["auc"] is None else float(eff_eval["auc"]),
            },
            "base_ensemble": {
                "threshold": float(base_threshold),
                "balanced_acc": float(base_eval["balanced_acc"]),
                "auc": None if base_eval["auc"] is None else float(base_eval["auc"]),
            },
        }

        torch.save(ckpt, best_path)
        print(
            f"Saved post-training thresholds | "
            f"Swin={swin_threshold:.2f} Eff={eff_threshold:.2f} Base50={base_threshold:.2f}"
        )

    summary_path = os.path.join(cfg["output_dir"], "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_auc": float(best_val_auc),
                "checkpoint": best_path,
                "base_ensemble_method": cfg["ensemble"]["base_method"],
                "initial_ensemble_weights": cfg["ensemble"]["initial_weights"],
                "post_training_weight_search": False,
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
