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
from post_training_ensemble_v3 import run_post_training_weight_search


def smooth_binary_labels(y, smoothing=0.0):
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing


def move_inputs_to_device(inputs, device):
    return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}


def eval_on_loader(model, loader, device, fake_idx):
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

    base_ensemble_probs = 0.5 * swin_probs + 0.5 * eff_probs
    base_threshold, _ = find_best_threshold(y_bin, base_ensemble_probs)

    return {
        "y_bin": y_bin,
        "swin_probs": swin_probs,
        "efficientnet_probs": eff_probs,
        "base_ensemble_probs": base_ensemble_probs,
        "base_threshold": base_threshold,
        "base_ensemble_eval": evaluate_predictions(y_bin, base_ensemble_probs, base_threshold),
        "swin_eval": evaluate_predictions(y_bin, swin_probs, base_threshold),
        "efficientnet_eval": evaluate_predictions(y_bin, eff_probs, base_threshold),
    }


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

    print("Using BCEWithLogitsLoss on 50/50 training ensemble logits")
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

        val_out = eval_on_loader(model, val_loader, device, fake_idx)
        base_eval = val_out["base_ensemble_eval"]
        swin_eval = val_out["swin_eval"]
        eff_eval = val_out["efficientnet_eval"]
        epoch_time = time.time() - epoch_start
        avg_loss = float(np.mean(losses)) if losses else float("nan")

        learned = run_post_training_weight_search(
            cfg=cfg,
            y_true=val_out["y_bin"],
            swin_probs=val_out["swin_probs"],
            eff_probs=val_out["efficientnet_probs"],
            output_dir=cfg["output_dir"],
            prefix=f"epoch_{epoch + 1:02d}_val",
        )
        learned_best = learned["best"]
        learned_eval = learned["final_eval"]

        print(
            f"epoch={epoch + 1} train_loss={avg_loss:.4f} epoch_time={epoch_time:.1f}s "
            f"base_val_bal_acc={base_eval['balanced_acc']:.4f} "
            f"learned_val_bal_acc={learned_eval['balanced_acc']:.4f} "
            f"best_w_swin={learned_best['swin_weight']:.2f} best_w_eff={learned_best['efficientnet_weight']:.2f} "
            f"best_t={learned_best['threshold']:.2f}"
            + (f" learned_val_auc={learned_eval['auc']:.4f}" if learned_eval["auc"] is not None else "")
        )
        print("Val true counts (real=0,fake=1):", learned_eval["true_counts"], "Val pred counts:", learned_eval["pred_counts"])
        print(
            f"Val component scores | 50/50 Ensemble bal_acc={base_eval['balanced_acc']:.4f} "
            f"Learned Ensemble bal_acc={learned_eval['balanced_acc']:.4f} "
            f"Swin bal_acc={swin_eval['balanced_acc']:.4f} "
            f"EfficientNet bal_acc={eff_eval['balanced_acc']:.4f}"
        )

        top_k = cfg["ensemble"].get("top_k_to_print", 10)
        print("Top searched ensemble weights this epoch:")
        for row in learned["ranked"][:top_k]:
            print(
                f"  swin={row['swin_weight']:.2f} eff={row['efficientnet_weight']:.2f} "
                f"thr={row['threshold']:.2f} bal_acc={row['balanced_acc']:.4f} "
                + (f"auc={row['auc']:.4f}" if row["auc"] is not None else "auc=None")
            )

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "epoch_time_sec": epoch_time,
            "base_ensemble_bal_acc": float(base_eval["balanced_acc"]),
            "learned_ensemble_bal_acc": float(learned_eval["balanced_acc"]),
            "swin_bal_acc": float(swin_eval["balanced_acc"]),
            "efficientnet_bal_acc": float(eff_eval["balanced_acc"]),
            "best_swin_weight": float(learned_best["swin_weight"]),
            "best_efficientnet_weight": float(learned_best["efficientnet_weight"]),
            "best_threshold": float(learned_best["threshold"]),
            "learned_ensemble_auc": None if learned_eval["auc"] is None else float(learned_eval["auc"]),
        })

        if learned_eval["balanced_acc"] > best_val_score:
            best_val_score = learned_eval["balanced_acc"]
            payload = {
                "model": model.state_dict(),
                "cfg": cfg,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "base_ensemble_method": cfg["ensemble"]["base_method"],
                "best_threshold": float(learned_best["threshold"]),
                "best_ensemble_weights": {
                    "swin": float(learned_best["swin_weight"]),
                    "efficientnet": float(learned_best["efficientnet_weight"]),
                },
                "val_score": float(learned_eval["balanced_acc"]),
                "val_auc": None if learned_eval["auc"] is None else float(learned_eval["auc"]),
                "swin_val_auc": None if swin_eval["auc"] is None else float(swin_eval["auc"]),
                "efficientnet_val_auc": None if eff_eval["auc"] is None else float(eff_eval["auc"]),
                "weight_search_json": learned["json_path"],
            }
            if cfg.get("save_val_outputs", True):
                payload["val_outputs"] = {
                    "y_bin": val_out["y_bin"].tolist(),
                    "swin_probs": val_out["swin_probs"].tolist(),
                    "efficientnet_probs": val_out["efficientnet_probs"].tolist(),
                    "base_ensemble_probs": val_out["base_ensemble_probs"].tolist(),
                    "learned_ensemble_probs": learned_best["probs"].tolist(),
                }
            torch.save(payload, best_path)
            print("Saved best checkpoint:", best_path)

    summary_path = os.path.join(cfg["output_dir"], "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_score": float(best_val_score),
                "checkpoint": best_path,
                "base_ensemble_method": cfg["ensemble"]["base_method"],
                "initial_ensemble_weights": cfg["ensemble"]["initial_weights"],
                "post_training_weight_search": cfg["ensemble"].get("learn_weights_post_training", False),
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
