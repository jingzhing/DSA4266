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


def eval_on_loader(model, loader, device, fake_idx, cfg, split_name="Validation"):
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

    best_t, best_score = find_best_threshold(
        y_bin,
        probs,
        metric=cfg["threshold_metric"],
        threshold_start=cfg["threshold_search"]["start"],
        threshold_end=cfg["threshold_search"]["end"],
        threshold_step=cfg["threshold_search"]["step"],
    )
    raw = evaluate_predictions(y_bin, probs, 0.5)
    searched = evaluate_predictions(y_bin, probs, best_t)
    summary = summarize_probabilities(probs, y_bin)
    mistakes = make_mistake_rows(all_paths, y_bin, probs, searched["pred"], top_k=cfg.get("diagnostics", {}).get("save_top_mistakes", 50)) if all_paths else []
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


def print_split_summary(name, out):
    raw = out["raw"]
    searched = out["searched"]
    summary = out["prob_summary"]
    print(
        f"{name} AUC={raw['auc']:.4f} | bal_acc@0.5={raw['balanced_acc']:.4f} | "
        f"bal_acc@best={searched['balanced_acc']:.4f} | best_t={out['best_t']:.2f}"
    )
    print(f"{name} true counts: {raw['true_counts']} | pred@0.5: {raw['pred_counts']} | pred@best: {searched['pred_counts']}")
    print(
        f"{name} prob means | real={summary['real']['mean']:.4f} fake={summary['fake']['mean']:.4f} "
        f"gap={summary['mean_gap_fake_minus_real']:.4f}"
    )


def save_diagnostics(cfg, split_name, out):
    out_dir = os.path.join(cfg["output_dir"], "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "split": split_name,
        "threshold": float(out["best_t"]),
        "raw": {
            "acc": float(out["raw"]["acc"]),
            "balanced_acc": float(out["raw"]["balanced_acc"]),
            "auc": None if out["raw"]["auc"] is None else float(out["raw"]["auc"]),
        },
        "searched": {
            "acc": float(out["searched"]["acc"]),
            "balanced_acc": float(out["searched"]["balanced_acc"]),
            "auc": None if out["searched"]["auc"] is None else float(out["searched"]["auc"]),
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
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    bundle = build_single_model_dataloaders(cfg)
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    classes = bundle["classes"]
    class_to_idx = bundle["class_to_idx"]
    fake_idx = class_to_idx["fake"]

    log_section("DATA SUMMARY")
    print("Device:", device)
    print("Classes:", classes)
    print("Mapping:", class_to_idx)
    print("Train split size:", len(bundle["train_ds"]))
    print("Val split size:", len(bundle["val_ds"]))
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

        val_out = eval_on_loader(model, val_loader, device, fake_idx, cfg, split_name="Validation")
        raw = val_out["raw"]
        searched = val_out["searched"]
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_time = time.time() - start
        current_auc = -1.0 if raw["auc"] is None else float(raw["auc"])

        if scheduler is not None:
            scheduler.step(current_auc)
            print("Current learning rates:", [group["lr"] for group in optimizer.param_groups])

        print(
            f"epoch={epoch + 1} train_loss={avg_loss:.4f} "
            f"val_auc={raw['auc']:.4f} val_bal_acc_0.5={raw['balanced_acc']:.4f} "
            f"val_bal_acc_best={searched['balanced_acc']:.4f} best_t={val_out['best_t']:.2f} "
            f"time={epoch_time:.1f}s"
        )
        print_split_summary("Validation", val_out)

        test_epoch_summary = None
        if cfg.get("diagnostics", {}).get("eval_test_each_epoch", False):
            test_out = eval_on_loader(model, test_loader, device, fake_idx, cfg, split_name="Test")
            print_split_summary("Test", test_out)
            gap = searched["balanced_acc"] - test_out["searched"]["balanced_acc"]
            print(f"Validation-Test gap at searched thresholds: {gap:.4f}")
            test_epoch_summary = {
                "test_auc": None if test_out["raw"]["auc"] is None else float(test_out["raw"]["auc"]),
                "test_bal_acc_at_0_5": float(test_out["raw"]["balanced_acc"]),
                "test_bal_acc_best": float(test_out["searched"]["balanced_acc"]),
                "test_best_threshold": float(test_out["best_t"]),
                "val_minus_test_bal_acc_best": float(gap),
            }

        history_row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_auc": None if raw["auc"] is None else float(raw["auc"]),
            "val_bal_acc_at_0_5": float(raw["balanced_acc"]),
            "val_bal_acc_best": float(searched["balanced_acc"]),
            "best_threshold": float(val_out["best_t"]),
            "epoch_time_sec": float(epoch_time),
        }
        if test_epoch_summary is not None:
            history_row.update(test_epoch_summary)
        history.append(history_row)

        if current_auc > best_val_auc + min_delta:
            best_val_auc = current_auc
            epochs_without_improve = 0
            payload = {
                "model": model.state_dict(),
                "cfg": cfg,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "best_threshold": float(val_out["best_t"]),
                "val_auc": None if raw["auc"] is None else float(raw["auc"]),
                "val_metrics_at_0_5": {
                    "acc": float(raw["acc"]),
                    "balanced_acc": float(raw["balanced_acc"]),
                    "auc": None if raw["auc"] is None else float(raw["auc"]),
                },
                "val_metrics_at_best_threshold": {
                    "acc": float(searched["acc"]),
                    "balanced_acc": float(searched["balanced_acc"]),
                    "auc": None if searched["auc"] is None else float(searched["auc"]),
                },
                "val_outputs": {
                    "y_bin": val_out["y_bin"].tolist(),
                    "paths": val_out["paths"],
                    "logits": val_out["logits"].tolist(),
                    "probs": val_out["probs"].tolist(),
                    "prob_summary": val_out["prob_summary"],
                },
            }
            torch.save(payload, best_path)
            print("Saved best checkpoint:", best_path)
            save_diagnostics(cfg, "val", val_out)
            if cfg.get("diagnostics", {}).get("eval_test_each_epoch", False):
                save_diagnostics(cfg, "test", test_out)
        else:
            epochs_without_improve += 1
            print(f"No significant improvement for {epochs_without_improve} epoch(s)")

        if epochs_without_improve >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    summary_path = os.path.join(cfg["output_dir"], "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_auc": float(best_val_auc), "history": history, "checkpoint": best_path}, f, indent=2)
    print("Saved training summary:", summary_path)


if __name__ == "__main__":
    main()
