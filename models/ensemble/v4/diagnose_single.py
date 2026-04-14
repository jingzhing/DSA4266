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
    if model_name in {"efficientnet", "effnet", "b2"}:
        return EFFICIENTNET_CONFIG
    raise ValueError("model_name must be one of: swin, efficientnet")


def unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
        paths = None
    else:
        x, y, paths = batch
    return x, y, paths


def run_split(model, loader, device, fake_idx, threshold, name, top_k):
    all_logits = []
    all_y = []
    all_paths = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=name):
            x, y, paths = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1).detach().cpu().numpy()
            all_logits.append(logits)
            all_y.append(y.numpy())
            all_paths.extend(list(paths))
    logits = np.concatenate(all_logits)
    y_orig = np.concatenate(all_y).astype(int)
    y_bin = (y_orig == fake_idx).astype(int)
    probs = sigmoid(logits)
    raw = evaluate_predictions(y_bin, probs, 0.5)
    searched = evaluate_predictions(y_bin, probs, threshold)
    summary = summarize_probabilities(probs, y_bin)
    mistakes = make_mistake_rows(all_paths, y_bin, probs, searched["pred"], top_k=top_k)
    return {
        "paths": all_paths,
        "y_bin": y_bin,
        "logits": logits,
        "probs": probs,
        "raw": raw,
        "searched": searched,
        "summary": summary,
        "mistakes": mistakes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["swin", "efficientnet"])
    args = parser.parse_args()

    cfg = get_cfg(args.model)
    init_run_logging(cfg["log_dir"], f"diagnose_{args.model}.log")
    log_section(f"DIAGNOSE {cfg['run_name']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    threshold = float(ckpt.get("best_threshold", cfg["default_threshold"]))

    bundle = build_single_model_dataloaders(cfg)
    class_to_idx = bundle["class_to_idx"]
    fake_idx = class_to_idx["fake"]

    model = build_binary_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_out = run_split(model, bundle["val_loader"], device, fake_idx, threshold, "Validation", cfg.get("diagnostics", {}).get("save_top_mistakes", 50))
    test_out = run_split(model, bundle["test_loader"], device, fake_idx, threshold, "Test", cfg.get("diagnostics", {}).get("save_top_mistakes", 50))

    print(f"Validation AUC={val_out['raw']['auc']:.4f} | bal_acc@0.5={val_out['raw']['balanced_acc']:.4f} | bal_acc@best={val_out['searched']['balanced_acc']:.4f}")
    print(f"Test       AUC={test_out['raw']['auc']:.4f} | bal_acc@0.5={test_out['raw']['balanced_acc']:.4f} | bal_acc@best={test_out['searched']['balanced_acc']:.4f}")
    print(f"Gap best-threshold balanced acc (val - test): {val_out['searched']['balanced_acc'] - test_out['searched']['balanced_acc']:.4f}")
    print(f"Mean fake prob gap | val={val_out['summary']['mean_gap_fake_minus_real']:.4f} | test={test_out['summary']['mean_gap_fake_minus_real']:.4f}")

    out_dir = os.path.join(cfg["output_dir"], "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"diagnose_{args.model}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "threshold_used": threshold,
            "validation": {
                "raw": {"acc": float(val_out['raw']['acc']), "balanced_acc": float(val_out['raw']['balanced_acc']), "auc": float(val_out['raw']['auc'])},
                "searched": {"acc": float(val_out['searched']['acc']), "balanced_acc": float(val_out['searched']['balanced_acc']), "auc": float(val_out['searched']['auc'])},
                "prob_summary": val_out['summary'],
            },
            "test": {
                "raw": {"acc": float(test_out['raw']['acc']), "balanced_acc": float(test_out['raw']['balanced_acc']), "auc": float(test_out['raw']['auc'])},
                "searched": {"acc": float(test_out['searched']['acc']), "balanced_acc": float(test_out['searched']['balanced_acc']), "auc": float(test_out['searched']['auc'])},
                "prob_summary": test_out['summary'],
            },
        }, f, indent=2)
    if val_out['mistakes']:
        save_rows_csv(os.path.join(out_dir, f"diagnose_{args.model}_val_top_mistakes.csv"), val_out['mistakes'])
    if test_out['mistakes']:
        save_rows_csv(os.path.join(out_dir, f"diagnose_{args.model}_test_top_mistakes.csv"), test_out['mistakes'])
    print("Saved diagnosis outputs to:", out_dir)


if __name__ == "__main__":
    main()
