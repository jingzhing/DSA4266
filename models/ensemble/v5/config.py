from copy import deepcopy
from pathlib import Path

RUN_TAG = "deepdetect_fresh_simpletf_seed777"
DATA_ROOT = Path(r"C:\Users\jz\DSA4266\DSA4266\data\deepdetect-2025_dddata\ddata")
SPLIT_SEED = 777

OUTPUT_ROOT = Path("data") / "outputs"
MODEL_ROOT = Path("data") / "models"
SPLIT_ROOT = OUTPUT_ROOT / "splits"

BASE_CONFIG = {
    "train_dir": str((DATA_ROOT / "train").resolve()),
    "test_dir": str((DATA_ROOT / "test").resolve()),
    "split_path": str((SPLIT_ROOT / f"{RUN_TAG}.json").resolve()),
    "seed": SPLIT_SEED,

    "batch_size": 16,
    "num_workers": 2,
    "pin_memory": True,

    "val_select_ratio": 0.10,
    "val_tune_ratio": 0.10,

    "threshold_metric": "balanced_acc",
    "default_threshold": 0.5,
    "threshold_search": {
        "start": 0.05,
        "end": 0.95,
        "step": 0.01,
    },

    "data_balance": {
        "use_weighted_sampler": False,
    },

    "augment": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.0,
        "color_jitter": 0.0,
        "random_erasing": 0.0,
    },

    "diagnostics": {
        "eval_test_each_epoch": False,
        "save_top_mistakes": 50,
    },

    "use_amp": True,
    "max_grad_norm": 1.0,
}

SWIN_CONFIG = deepcopy(BASE_CONFIG)
SWIN_CONFIG.update({
    "run_name": f"swin_{RUN_TAG}",
    "checkpoint_dir": str((MODEL_ROOT / "swin" / RUN_TAG / "checkpoints").resolve()),
    "output_dir": str((OUTPUT_ROOT / f"swin_{RUN_TAG}").resolve()),
    "log_dir": str((OUTPUT_ROOT / f"swin_{RUN_TAG}" / "logs").resolve()),
    "train_log_name": "train.log",
    "test_log_name": "test.log",
    "model": {
        "arch": "swin_tiny_patch4_window7_224",
        "img_size": 224,
        "dropout": 0.2,
        "drop_path_rate": 0.1,
        "freeze_backbone": False,
        "partial_finetune": False,
    },
    "train": {
        "epochs": 18,
        "lr": 1e-4,
        "weight_decay": 5e-5,
        "label_smoothing": 0.03,
        "optimizer": "adamw",
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 5e-4,
        "scheduler": {
            "name": "reduce_on_plateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 1,
            "min_lr": 1e-6,
        },
    },
})

EFFICIENTNET_CONFIG = deepcopy(BASE_CONFIG)
EFFICIENTNET_CONFIG.update({
    "run_name": f"efficientnet_b3_{RUN_TAG}",
    "checkpoint_dir": str((MODEL_ROOT / "efficientnet_b3" / RUN_TAG / "checkpoints").resolve()),
    "output_dir": str((OUTPUT_ROOT / f"efficientnet_b3_{RUN_TAG}").resolve()),
    "log_dir": str((OUTPUT_ROOT / f"efficientnet_b3_{RUN_TAG}" / "logs").resolve()),
    "train_log_name": "train.log",
    "test_log_name": "test.log",
    "model": {
        "arch": "efficientnet_b3",
        "img_size": 300,
        "dropout": 0.2,
        "drop_path_rate": 0.0,
        "freeze_backbone": False,
        "partial_finetune": True,
        "partial_finetune_blocks": 30,
    },
    "train": {
        "epochs": 15,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "optimizer": "adamw",
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 5e-4,
        "scheduler": {
            "name": "reduce_on_plateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 1,
            "min_lr": 1e-6,
        },
    },
})

ENSEMBLE_CONFIG = deepcopy(BASE_CONFIG)
ENSEMBLE_CONFIG.update({
    "run_name": f"ensemble_posthoc_{RUN_TAG}",
    "output_dir": str((OUTPUT_ROOT / f"ensemble_posthoc_{RUN_TAG}").resolve()),
    "log_dir": str((OUTPUT_ROOT / f"ensemble_posthoc_{RUN_TAG}" / "logs").resolve()),
    "search_on": "logits",
    "weight_search": {
        "start": 0.0,
        "end": 1.0,
        "step": 0.02,
    },
    "save_test_outputs": True,
    "swin_checkpoint": str((MODEL_ROOT / "swin" / RUN_TAG / "checkpoints" / "best.pt").resolve()),
    "efficientnet_checkpoint": str((MODEL_ROOT / "efficientnet_b3" / RUN_TAG / "checkpoints" / "best.pt").resolve()),
})
