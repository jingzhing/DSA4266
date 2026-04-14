import os
import numpy as np
import torch

from config import SWIN_CONFIG, EFFICIENTNET_CONFIG, ENSEMBLE_CONFIG
from data import build_single_model_dataloaders
from logging_utils import init_run_logging, log_section
from metrics import sigmoid, evaluate_predictions, search_best_weight, combine_logits, combine_probabilities, save_json
from models import build_binary_model


def run_model_on_loader(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for batch in loader:
            x, y, _ = batch
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1).detach().cpu().numpy()
            all_logits.append(logits)
            all_y.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_y).astype(int)


def load_model_from_ckpt(cfg, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_binary_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main():
    cfg = ENSEMBLE_CONFIG
    init_run_logging(cfg["log_dir"], "ensemble_posthoc.log")
    log_section("POST-HOC ENSEMBLE SEARCH")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    swin_bundle = build_single_model_dataloaders(SWIN_CONFIG)
    eff_bundle = build_single_model_dataloaders(EFFICIENTNET_CONFIG)

    if swin_bundle["class_to_idx"] != eff_bundle["class_to_idx"]:
        raise RuntimeError("Swin and EfficientNet class mappings do not match")
    class_to_idx = swin_bundle["class_to_idx"]
    fake_idx = class_to_idx["fake"]

    if list(swin_bundle["base_val_tune_ds"].indices) != list(eff_bundle["base_val_tune_ds"].indices):
        raise RuntimeError("Val_tune splits do not match. Use the same shared split file.")
    if SWIN_CONFIG["train_dir"] != EFFICIENTNET_CONFIG["train_dir"] or SWIN_CONFIG["test_dir"] != EFFICIENTNET_CONFIG["test_dir"]:
        raise RuntimeError("Train/test directories do not match across solo models")

    swin_model, swin_ckpt = load_model_from_ckpt(SWIN_CONFIG, cfg["swin_checkpoint"], device)
    eff_model, eff_ckpt = load_model_from_ckpt(EFFICIENTNET_CONFIG, cfg["efficientnet_checkpoint"], device)

    log_section("VAL_TUNE FOR ENSEMBLE SEARCH")
    swin_val_logits, y_val_orig = run_model_on_loader(swin_model, swin_bundle["val_tune_loader"], device)
    eff_val_logits, y_val_orig_eff = run_model_on_loader(eff_model, eff_bundle["val_tune_loader"], device)
    if not np.array_equal(y_val_orig, y_val_orig_eff):
        raise RuntimeError("Val_tune labels do not match between models")
    y_val = (y_val_orig == fake_idx).astype(int)
    swin_val_probs = sigmoid(swin_val_logits)
    eff_val_probs = sigmoid(eff_val_logits)

    best, ranked = search_best_weight(
        y_true=y_val,
        logits_a=swin_val_logits,
        logits_b=eff_val_logits,
        probs_a=swin_val_probs,
        probs_b=eff_val_probs,
        search_on=cfg["search_on"],
        metric=cfg["threshold_metric"],
        weight_start=cfg["weight_search"]["start"],
        weight_end=cfg["weight_search"]["end"],
        weight_step=cfg["weight_search"]["step"],
        threshold_start=cfg["threshold_search"]["start"],
        threshold_end=cfg["threshold_search"]["end"],
        threshold_step=cfg["threshold_search"]["step"],
    )

    if cfg["search_on"] == "logits":
        val_fused_probs = sigmoid(combine_logits(swin_val_logits, eff_val_logits, best["weight_a"]))
    else:
        val_fused_probs = combine_probabilities(swin_val_probs, eff_val_probs, best["weight_a"])
    val_eval = evaluate_predictions(y_val, val_fused_probs, best["threshold"])

    print("Best val_tune ensemble:")
    print(best)

    log_section("TEST WITH FROZEN ENSEMBLE SETTINGS")
    swin_test_logits, y_test_orig = run_model_on_loader(swin_model, swin_bundle["test_loader"], device)
    eff_test_logits, y_test_orig_eff = run_model_on_loader(eff_model, eff_bundle["test_loader"], device)
    if not np.array_equal(y_test_orig, y_test_orig_eff):
        raise RuntimeError("Test labels do not match between models")
    y_test = (y_test_orig == fake_idx).astype(int)
    swin_test_probs = sigmoid(swin_test_logits)
    eff_test_probs = sigmoid(eff_test_logits)

    if cfg["search_on"] == "logits":
        test_fused_probs = sigmoid(combine_logits(swin_test_logits, eff_test_logits, best["weight_a"]))
    else:
        test_fused_probs = combine_probabilities(swin_test_probs, eff_test_probs, best["weight_a"])

    raw_swin = evaluate_predictions(y_test, swin_test_probs, 0.5)
    raw_eff = evaluate_predictions(y_test, eff_test_probs, 0.5)
    raw_ens = evaluate_predictions(y_test, test_fused_probs, 0.5)
    searched_swin = evaluate_predictions(y_test, swin_test_probs, float(swin_ckpt.get("best_threshold", 0.5)))
    searched_eff = evaluate_predictions(y_test, eff_test_probs, float(eff_ckpt.get("best_threshold", 0.5)))
    searched_ens = evaluate_predictions(y_test, test_fused_probs, best["threshold"])

    print("SWIN test @ best threshold:", float(swin_ckpt.get("best_threshold", 0.5)))
    print(f"Acc={searched_swin['acc']:.4f} | Balanced Acc={searched_swin['balanced_acc']:.4f} | AUC={searched_swin['auc']:.4f}")
    print("EFFICIENTNET test @ best threshold:", float(eff_ckpt.get("best_threshold", 0.5)))
    print(f"Acc={searched_eff['acc']:.4f} | Balanced Acc={searched_eff['balanced_acc']:.4f} | AUC={searched_eff['auc']:.4f}")
    print("ENSEMBLE test @ frozen val_tune threshold:", best["threshold"])
    print(f"Acc={searched_ens['acc']:.4f} | Balanced Acc={searched_ens['balanced_acc']:.4f} | AUC={searched_ens['auc']:.4f}")
    print("Ensemble confusion matrix:")
    print(searched_ens["cm"])
    print("Ensemble report:")
    print(searched_ens["report"])

    out = {
        "search_on": cfg["search_on"],
        "best_val_tune_ensemble": best,
        "val_tune_metrics": {
            "acc": float(val_eval["acc"]),
            "balanced_acc": float(val_eval["balanced_acc"]),
            "auc": None if val_eval["auc"] is None else float(val_eval["auc"]),
        },
        "test_metrics": {
            "swin_at_0_5": {
                "acc": float(raw_swin["acc"]),
                "balanced_acc": float(raw_swin["balanced_acc"]),
                "auc": None if raw_swin["auc"] is None else float(raw_swin["auc"]),
            },
            "efficientnet_at_0_5": {
                "acc": float(raw_eff["acc"]),
                "balanced_acc": float(raw_eff["balanced_acc"]),
                "auc": None if raw_eff["auc"] is None else float(raw_eff["auc"]),
            },
            "ensemble_at_0_5": {
                "acc": float(raw_ens["acc"]),
                "balanced_acc": float(raw_ens["balanced_acc"]),
                "auc": None if raw_ens["auc"] is None else float(raw_ens["auc"]),
            },
            "swin_at_best_threshold": {
                "threshold": float(swin_ckpt.get("best_threshold", 0.5)),
                "acc": float(searched_swin["acc"]),
                "balanced_acc": float(searched_swin["balanced_acc"]),
                "auc": None if searched_swin["auc"] is None else float(searched_swin["auc"]),
            },
            "efficientnet_at_best_threshold": {
                "threshold": float(eff_ckpt.get("best_threshold", 0.5)),
                "acc": float(searched_eff["acc"]),
                "balanced_acc": float(searched_eff["balanced_acc"]),
                "auc": None if searched_eff["auc"] is None else float(searched_eff["auc"]),
            },
            "ensemble_at_frozen_threshold": {
                "threshold": float(best["threshold"]),
                "acc": float(searched_ens["acc"]),
                "balanced_acc": float(searched_ens["balanced_acc"]),
                "auc": None if searched_ens["auc"] is None else float(searched_ens["auc"]),
            },
        },
        "top_ranked": ranked[:20],
    }

    if cfg.get("save_test_outputs", True):
        out["outputs"] = {
            "y_test": y_test.tolist(),
            "swin_test_logits": swin_test_logits.tolist(),
            "eff_test_logits": eff_test_logits.tolist(),
            "swin_test_probs": swin_test_probs.tolist(),
            "eff_test_probs": eff_test_probs.tolist(),
            "ensemble_test_probs": test_fused_probs.tolist(),
        }

    save_json(os.path.join(cfg["output_dir"], "ensemble_summary.json"), out)
    print("Saved ensemble summary:", os.path.join(cfg["output_dir"], "ensemble_summary.json"))


if __name__ == "__main__":
    main()
