import argparse
import json
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from config import SWIN_CONFIG, EFFICIENTNET_CONFIG
from data import build_single_model_dataloaders, set_seed
from logging_utils import init_run_logging, log_section
from metrics import sigmoid, find_best_threshold, evaluate_predictions, summarize_probabilities, make_mistake_rows, save_rows_csv
from models import build_binary_model, build_optimizer, build_scheduler


def smooth_binary_labels(y, smoothing=0.0):
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing


def get_cfg(model_name):
    model_name = model_name.lower()
    if model_name == "swin":
        return SWIN_CONFIG
    if model_name in {"efficientnet", "effnet", "b3"}:
        return EFFICIENTNET_CONFIG
    raise ValueError("model_name must be one of: swin, efficientnet")


def unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
        paths = None
    elif len(batch) == 3:
        x, y, paths = batch
    else:
        raise ValueError("Unexpected batch format")
    return x, y, paths


def eval_on_loader(model, loader, device, fake_idx, threshold, split_name="Validation"):
    model.eval()
    all_logits = []
    all_y = []
    all_paths = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=split_name, leave=False):
            x, y, paths = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1).detach().cpu().numpy()
            all_logits.append(logits)
            all_y.append(y.numpy())
            if paths is not None:
                all_paths.extend(list(paths))

    logits = np.concatenate(all_logits)
    y_orig = np.concatenate(all_y).astype(int)
    y_bin = (y_orig == fake_idx).astype(int)
    probs = sigmoid(logits)
    raw = evaluate_predictions(y_bin, probs, 0.5)
    chosen = evaluate_predictions(y_bin, probs, threshold)
    summary = summarize_probabilities(probs, y_bin)
    mistakes = make_mistake_rows(all_paths, y_bin, probs, chosen["pred"], top_k=50) if all_paths else []
    return {
        "logits": logits,
        "probs": probs,
        "y_bin": y_bin,
        "paths": all_paths,
        "raw": raw,
        "chosen": chosen,
        "prob_summary": summary,
        "mistakes": mistakes,
    }


def save_diagnostics(cfg, split_name, out, threshold):
    out_dir = os.path.join(cfg["output_dir"], "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "split": split_name,
        "threshold": float(threshold),
        "raw": {
            "acc": float(out["raw"]["acc"]),
            "balanced_acc": float(out["raw"]["balanced_acc"]),
            "auc": None if out["raw"]["auc"] is None else float(out["raw"]["auc"]),
        },
        "chosen": {
            "acc": float(out["chosen"]["acc"]),
            "balanced_acc": float(out["chosen"]["balanced_acc"]),
            "auc": None if out["chosen"]["auc"] is None else float(out["chosen"]["auc"]),
        },
        "prob_summary": out["prob_summary"],
    }
    with open(os.path.join(out_dir, f"{split_name.lower()}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if out["mistakes"]:
        save_rows_csv(os.path.join(out_dir, f"{split_name.lower()}_top_mistakes.csv"), out["mistakes"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["swin", "efficientnet"])
    args = parser.parse_args()

    cfg = get_cfg(args.model)
    set_seed(cfg["seed"])
    init_run_logging(cfg["log_dir"], cfg["train_log_name"])
    log_section(f"TRAIN {cfg['run_name']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg.get("use_amp", True) and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    bundle = build_single_model_dataloaders(cfg)
    train_loader = bundle["train_loader"]
    val_select_loader = bundle["val_select_loader"]
    val_tune_loader = bundle["val_tune_loader"]
    classes = bundle["classes"]
    class_to_idx = bundle["class_to_idx"]
    fake_idx = class_to_idx["fake"]

    log_section("DATA SUMMARY")
    print("Device:", device)
    print("Train dir:", cfg["train_dir"])
    print("Test dir:", cfg["test_dir"])
    print("Classes:", classes)
    print("Mapping:", class_to_idx)
    print("Train split size:", len(bundle["train_ds"]))
    print("Val_select size:", len(bundle["val_select_ds"]))
    print("Val_tune size:", len(bundle["val_tune_ds"]))
    print("Test size:", len(bundle["test_ds"]))
    print("Model config:", cfg["model"])
    print("Train config:", cfg["train"])

    model = build_binary_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
    history = []
    patience = int(cfg["train"]["early_stopping_patience"])
    min_delta = float(cfg["train"]["early_stopping_min_delta"])
    epochs_without_improve = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        losses = []
        start = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']} Train"):
            x, y, _ = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y_bin = (y == fake_idx).float().to(device, non_blocking=True)
            y_sm = smooth_binary_labels(y_bin, cfg["train"].get("label_smoothing", 0.0))

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x).squeeze(1)
                loss = loss_fn(logits, y_sm)
            scaler.scale(loss).backward()
            if cfg.get("max_grad_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        # val_select only for checkpoint selection at fixed threshold 0.5
        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for batch in tqdm(val_select_loader, desc="Val_select", leave=False):
                x, y, _ = unpack_batch(batch)
                x = x.to(device, non_blocking=True)
                logits = model(x).squeeze(1).detach().cpu().numpy()
                all_logits.append(logits)
                all_y.append(y.numpy())
        val_select_logits = np.concatenate(all_logits)
        val_select_y = np.concatenate(all_y).astype(int)
        val_select_y = (val_select_y == fake_idx).astype(int)
        val_select_probs = sigmoid(val_select_logits)
        val_select_raw = evaluate_predictions(val_select_y, val_select_probs, 0.5)

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_time = time.time() - start
        current_auc = -1.0 if val_select_raw["auc"] is None else float(val_select_raw["auc"])

        if scheduler is not None:
            scheduler.step(current_auc)
            print("Current learning rates:", [group["lr"] for group in optimizer.param_groups])

        print(
            f"epoch={epoch + 1} train_loss={avg_loss:.4f} "
            f"val_select_auc={val_select_raw['auc']:.4f} val_select_bal_acc_0.5={val_select_raw['balanced_acc']:.4f} "
            f"time={epoch_time:.1f}s"
        )

        history_row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_select_auc": None if val_select_raw["auc"] is None else float(val_select_raw["auc"]),
            "val_select_bal_acc_at_0_5": float(val_select_raw["balanced_acc"]),
            "epoch_time_sec": float(epoch_time),
        }
        history.append(history_row)

        if current_auc > best_val_auc + min_delta:
            best_val_auc = current_auc
            epochs_without_improve = 0
            payload = {
                "model": model.state_dict(),
                "cfg": cfg,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "best_threshold": float(cfg["default_threshold"]),
                "val_select_metrics_at_0_5": {
                    "acc": float(val_select_raw["acc"]),
                    "balanced_acc": float(val_select_raw["balanced_acc"]),
                    "auc": None if val_select_raw["auc"] is None else float(val_select_raw["auc"]),
                },
            }
            torch.save(payload, best_path)
            print("Saved best checkpoint:", best_path)
        else:
            epochs_without_improve += 1
            print(f"No significant improvement for {epochs_without_improve} epoch(s)")

        if epochs_without_improve >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    if not os.path.exists(best_path):
        raise RuntimeError(f"Best checkpoint was not saved: {best_path}")

    # one-time threshold tuning on val_tune using the frozen best checkpoint
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    tune_out = eval_on_loader(model, val_tune_loader, device, fake_idx, 0.5, split_name="Val_tune")
    best_t, best_score = find_best_threshold(
        tune_out["y_bin"],
        tune_out["probs"],
        metric=cfg["threshold_metric"],
        threshold_start=cfg["threshold_search"]["start"],
        threshold_end=cfg["threshold_search"]["end"],
        threshold_step=cfg["threshold_search"]["step"],
    )
    tune_eval = evaluate_predictions(tune_out["y_bin"], tune_out["probs"], best_t)
    tune_out["chosen"] = tune_eval
    print(
        f"Val_tune best threshold={best_t:.2f} | score={best_score:.4f} | "
        f"acc={tune_eval['acc']:.4f} | bal_acc={tune_eval['balanced_acc']:.4f} | auc={tune_eval['auc']:.4f}"
    )

    ckpt["best_threshold"] = float(best_t)
    ckpt["val_tune_metrics_at_best_threshold"] = {
        "acc": float(tune_eval["acc"]),
        "balanced_acc": float(tune_eval["balanced_acc"]),
        "auc": None if tune_eval["auc"] is None else float(tune_eval["auc"]),
    }
    ckpt["val_tune_outputs"] = {
        "y_bin": tune_out["y_bin"].tolist(),
        "paths": tune_out["paths"],
        "logits": tune_out["logits"].tolist(),
        "probs": tune_out["probs"].tolist(),
        "prob_summary": tune_out["prob_summary"],
    }
    torch.save(ckpt, best_path)
    save_diagnostics(cfg, "val_tune", tune_out, best_t)

    summary_path = os.path.join(cfg["output_dir"], "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_select_auc": float(best_val_auc), "best_threshold": float(best_t), "history": history, "checkpoint": best_path}, f, indent=2)
    print("Saved training summary:", summary_path)


if __name__ == "__main__":
    main()
