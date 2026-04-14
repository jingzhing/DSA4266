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
    if model_name in {"efficientnet", "effnet", "b2"}:
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


def eval_on_loader(model, loader, device, fake_idx, cfg, split_name="Validation", search_threshold=False):
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
    best_t = cfg["default_threshold"]
    best_score = None
    searched = None
    mistakes = []

    if search_threshold:
        best_t, best_score = find_best_threshold(
            y_bin,
            probs,
            metric=cfg["threshold_metric"],
            threshold_start=cfg["threshold_search"]["start"],
            threshold_end=cfg["threshold_search"]["end"],
            threshold_step=cfg["threshold_search"]["step"],
        )
        searched = evaluate_predictions(y_bin, probs, best_t)
        mistakes = make_mistake_rows(
            all_paths,
            y_bin,
            probs,
            searched["pred"],
            top_k=cfg.get("diagnostics", {}).get("save_top_mistakes", 50),
        ) if all_paths else []

    summary = summarize_probabilities(probs, y_bin)
    return {
        "logits": logits,
        "probs": probs,
        "y_bin": y_bin,
        "paths": all_paths,
        "raw": raw,
        "searched": searched,
        "best_t": best_t,
        "best_score": best_score,
        "prob_summary": summary,
        "mistakes": mistakes,
    }


def print_split_summary(name, out, use_searched=False):
    raw = out["raw"]
    summary = out["prob_summary"]
    if use_searched and out["searched"] is not None:
        searched = out["searched"]
        print(
            f"{name} AUC={raw['auc']:.4f} | bal_acc@0.5={raw['balanced_acc']:.4f} | "
            f"bal_acc@best={searched['balanced_acc']:.4f} | best_t={out['best_t']:.2f}"
        )
        print(
            f"{name} true counts: {raw['true_counts']} | pred@0.5: {raw['pred_counts']} | "
            f"pred@best: {searched['pred_counts']}"
        )
    else:
        print(
            f"{name} AUC={raw['auc']:.4f} | bal_acc@0.5={raw['balanced_acc']:.4f} | "
            f"acc@0.5={raw['acc']:.4f}"
        )
        print(f"{name} true counts: {raw['true_counts']} | pred@0.5: {raw['pred_counts']}")
    print(
        f"{name} prob means | real={summary['real']['mean']:.4f} fake={summary['fake']['mean']:.4f} "
        f"gap={summary['mean_gap_fake_minus_real']:.4f}"
    )


def save_diagnostics(cfg, split_name, out, use_searched=False):
    out_dir = os.path.join(cfg["output_dir"], "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "split": split_name,
        "raw": {
            "acc": float(out["raw"]["acc"]),
            "balanced_acc": float(out["raw"]["balanced_acc"]),
            "auc": None if out["raw"]["auc"] is None else float(out["raw"]["auc"]),
        },
        "prob_summary": out["prob_summary"],
    }
    if use_searched and out["searched"] is not None:
        payload["threshold"] = float(out["best_t"])
        payload["searched"] = {
            "acc": float(out["searched"]["acc"]),
            "balanced_acc": float(out["searched"]["balanced_acc"]),
            "auc": None if out["searched"]["auc"] is None else float(out["searched"]["auc"]),
        }
    with open(os.path.join(out_dir, f"{split_name.lower()}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if use_searched and out["mistakes"]:
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
    print("Classes:", classes)
    print("Mapping:", class_to_idx)
    print("Train split size:", len(bundle["train_ds"]))
    print("Val-select split size:", len(bundle["val_select_ds"]))
    print("Val-tune split size:", len(bundle["val_tune_ds"]))
    print("Test size:", len(bundle["test_ds"]))
    print("Model config:", cfg["model"])
    print("Train config:", cfg["train"])
    print("Diagnostics config:", cfg.get("diagnostics", {}))

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
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y_sm)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_select_out = eval_on_loader(
            model,
            val_select_loader,
            device,
            fake_idx,
            cfg,
            split_name="Val-Select",
            search_threshold=False,
        )
        raw = val_select_out["raw"]
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_time = time.time() - start
        current_auc = -1.0 if raw["auc"] is None else float(raw["auc"])

        if scheduler is not None:
            scheduler.step(current_auc)
            print("Current learning rates:", [group["lr"] for group in optimizer.param_groups])

        print(
            f"epoch={epoch + 1} train_loss={avg_loss:.4f} "
            f"val_select_auc={raw['auc']:.4f} val_select_bal_acc_0.5={raw['balanced_acc']:.4f} "
            f"time={epoch_time:.1f}s"
        )
        print_split_summary("Val-Select", val_select_out, use_searched=False)

        history_row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_select_auc": None if raw["auc"] is None else float(raw["auc"]),
            "val_select_bal_acc_at_0_5": float(raw["balanced_acc"]),
            "val_select_acc_at_0_5": float(raw["acc"]),
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
                "selection_split": "val_select",
                "tuning_split": "val_tune",
                "val_select_metrics_at_0_5": {
                    "acc": float(raw["acc"]),
                    "balanced_acc": float(raw["balanced_acc"]),
                    "auc": None if raw["auc"] is None else float(raw["auc"]),
                },
                "val_select_outputs": {
                    "y_bin": val_select_out["y_bin"].tolist(),
                    "paths": val_select_out["paths"],
                    "logits": val_select_out["logits"].tolist(),
                    "probs": val_select_out["probs"].tolist(),
                    "prob_summary": val_select_out["prob_summary"],
                },
            }
            torch.save(payload, best_path)
            print("Saved best checkpoint:", best_path)
            save_diagnostics(cfg, "val_select", val_select_out, use_searched=False)
        else:
            epochs_without_improve += 1
            print(f"No significant improvement for {epochs_without_improve} epoch(s)")

        if epochs_without_improve >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    log_section("POST-TRAIN THRESHOLD TUNING ON VAL-TUNE")
    best_ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model"])
    model.to(device)
    model.eval()

    val_tune_out = eval_on_loader(
        model,
        val_tune_loader,
        device,
        fake_idx,
        cfg,
        split_name="Val-Tune",
        search_threshold=True,
    )
    print_split_summary("Val-Tune", val_tune_out, use_searched=True)
    save_diagnostics(cfg, "val_tune", val_tune_out, use_searched=True)

    best_ckpt["best_threshold"] = float(val_tune_out["best_t"])
    best_ckpt["threshold_metric"] = cfg["threshold_metric"]
    best_ckpt["threshold_tuning_split"] = "val_tune"
    best_ckpt["val_tune_metrics_at_0_5"] = {
        "acc": float(val_tune_out["raw"]["acc"]),
        "balanced_acc": float(val_tune_out["raw"]["balanced_acc"]),
        "auc": None if val_tune_out["raw"]["auc"] is None else float(val_tune_out["raw"]["auc"]),
    }
    best_ckpt["val_tune_metrics_at_best_threshold"] = {
        "acc": float(val_tune_out["searched"]["acc"]),
        "balanced_acc": float(val_tune_out["searched"]["balanced_acc"]),
        "auc": None if val_tune_out["searched"]["auc"] is None else float(val_tune_out["searched"]["auc"]),
    }
    best_ckpt["val_tune_outputs"] = {
        "y_bin": val_tune_out["y_bin"].tolist(),
        "paths": val_tune_out["paths"],
        "logits": val_tune_out["logits"].tolist(),
        "probs": val_tune_out["probs"].tolist(),
        "prob_summary": val_tune_out["prob_summary"],
    }
    torch.save(best_ckpt, best_path)
    print("Updated checkpoint with frozen threshold:", best_path)

    summary_path = os.path.join(cfg["output_dir"], "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_val_select_auc": float(best_val_auc),
            "frozen_best_threshold": float(val_tune_out["best_t"]),
            "history": history,
            "checkpoint": best_path,
        }, f, indent=2)
    print("Saved training summary:", summary_path)


if __name__ == "__main__":
    main()
