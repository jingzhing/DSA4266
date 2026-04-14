import json
import os

from ensemble_metrics_v3 import evaluate_predictions, search_best_ensemble_weight


def run_post_training_weight_search(cfg, y_true, swin_probs, eff_probs, output_dir, prefix="val"):
    ensemble_cfg = cfg["ensemble"]
    threshold_cfg = cfg.get("threshold_search", {})

    best, ranked = search_best_ensemble_weight(
        y_true=y_true,
        swin_probs=swin_probs,
        eff_probs=eff_probs,
        metric=cfg["threshold_metric"],
        weight_start=ensemble_cfg.get("weight_search_start", 0.0),
        weight_end=ensemble_cfg.get("weight_search_end", 1.0),
        weight_step=ensemble_cfg.get("weight_search_step", 0.02),
        threshold_start=threshold_cfg.get("start", 0.01),
        threshold_end=threshold_cfg.get("end", 0.99),
        threshold_step=threshold_cfg.get("step", 0.01),
    )

    final_eval = evaluate_predictions(y_true, best["probs"], best["threshold"])

    serializable_ranked = []
    for row in ranked:
        serializable_ranked.append({
            "swin_weight": row["swin_weight"],
            "efficientnet_weight": row["efficientnet_weight"],
            "threshold": row["threshold"],
            "score": row["score"],
            "balanced_acc": row["balanced_acc"],
            "acc": row["acc"],
            "auc": row["auc"],
        })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}_ensemble_weight_search.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best": {
                    "swin_weight": best["swin_weight"],
                    "efficientnet_weight": best["efficientnet_weight"],
                    "threshold": best["threshold"],
                    "score": best["score"],
                    "balanced_acc": final_eval["balanced_acc"],
                    "acc": final_eval["acc"],
                    "auc": final_eval["auc"],
                },
                "top_ranked": serializable_ranked,
            },
            f,
            indent=2,
        )

    return {
        "best": best,
        "ranked": ranked,
        "final_eval": final_eval,
        "json_path": out_path,
    }
