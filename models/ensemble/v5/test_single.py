import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm

from config import SWIN_CONFIG, EFFICIENTNET_CONFIG
from data import build_single_model_dataloaders
from logging_utils import init_run_logging, log_section
from metrics import sigmoid, evaluate_predictions, summarize_probabilities, make_mistake_rows, save_rows_csv
from models import build_binary_model


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["swin", "efficientnet"])
    args = parser.parse_args()

    cfg = get_cfg(args.model)
    init_run_logging(cfg["log_dir"], cfg["test_log_name"])
    log_section(f"TEST {cfg['run_name']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    bundle = build_single_model_dataloaders(cfg)
    test_loader = bundle["test_loader"]
    test_ds = bundle["test_ds"]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_t = float(ckpt.get("best_threshold", cfg["default_threshold"]))
    class_to_idx = ckpt.get("class_to_idx", test_ds.class_to_idx)
    if class_to_idx != test_ds.class_to_idx:
        raise RuntimeError(f"Class mapping mismatch. ckpt={class_to_idx} test={test_ds.class_to_idx}")

    fake_idx = class_to_idx["fake"]
    model = build_binary_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_logits = []
    all_y = []
    all_paths = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
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
    searched = evaluate_predictions(y_bin, probs, best_t)
    summary = summarize_probabilities(probs, y_bin)
    mistakes = make_mistake_rows(all_paths, y_bin, probs, searched["pred"], top_k=cfg.get("diagnostics", {}).get("save_top_mistakes", 50))

    print("Mapping:", class_to_idx)
    print("Best threshold:", best_t)
    print(f"Prob means | real={summary['real']['mean']:.4f} fake={summary['fake']['mean']:.4f} gap={summary['mean_gap_fake_minus_real']:.4f}")
    print("RAW @ 0.50")
    print(raw["cm"])
    print(raw["report"])
    print(f"Acc={raw['acc']:.4f} | Balanced Acc={raw['balanced_acc']:.4f} | AUC={raw['auc']:.4f}")
    print("\nSEARCHED THRESHOLD")
    print(searched["cm"])
    print(searched["report"])
    print(f"Acc={searched['acc']:.4f} | Balanced Acc={searched['balanced_acc']:.4f} | AUC={searched['auc']:.4f}")

    out_dir = os.path.join(cfg["output_dir"], "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "threshold": float(best_t),
        "raw": {
            "acc": float(raw["acc"]),
            "balanced_acc": float(raw["balanced_acc"]),
            "auc": None if raw["auc"] is None else float(raw["auc"]),
        },
        "searched": {
            "acc": float(searched["acc"]),
            "balanced_acc": float(searched["balanced_acc"]),
            "auc": None if searched["auc"] is None else float(searched["auc"]),
        },
        "prob_summary": summary,
        "n_samples": int(len(y_bin)),
    }
    with open(os.path.join(out_dir, "test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if mistakes:
        save_rows_csv(os.path.join(out_dir, "test_top_mistakes.csv"), mistakes)
    with open(os.path.join(out_dir, "test_outputs.json"), "w", encoding="utf-8") as f:
        json.dump({
            "paths": all_paths,
            "y_bin": y_bin.tolist(),
            "logits": logits.tolist(),
            "probs": probs.tolist(),
        }, f)
    print("Saved diagnostics to:", out_dir)


if __name__ == "__main__":
    main()
