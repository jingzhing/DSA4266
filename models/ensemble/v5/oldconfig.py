from copy import deepcopy
from pathlib import Path
import os


THIS_DIR = Path(__file__).resolve().parent


def _find_data_root() -> Path:
    env = os.environ.get("DEEPDETECT_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    candidates = []
    for base in [THIS_DIR, *THIS_DIR.parents]:
        candidates.extend([
            base / "data" / "deepdetect-2025" / "ddata",
            base / "deepdetect-2025" / "ddata",
        ])

    for p in candidates:
        if (p / "train").exists() and (p / "test").exists():
            return p.resolve()

    return (THIS_DIR / "data" / "deepdetect-2025" / "ddata").resolve()


def _repo_root() -> Path:
    data_root = _find_data_root()
    if data_root.parent.name == "deepdetect-2025":
        maybe_root = data_root.parent.parent
        if maybe_root.exists():
            return maybe_root.resolve()
    return THIS_DIR.resolve()


DATA_ROOT = _find_data_root()
REPO_ROOT = _repo_root()
OUTPUT_ROOT = REPO_ROOT / "outputs"
MODEL_ROOT = REPO_ROOT / "models"

BASE_CONFIG = {
    "train_dir": str((DATA_ROOT / "train").resolve()),
    "test_dir": str((DATA_ROOT / "test").resolve()),
    "split_path": str((OUTPUT_ROOT / "deepdetect" / "fixed_split_seed42.json").resolve()),
    "seed": 42,
    "batch_size": 16,
    "num_workers": 2,
    "pin_memory": True,
    "use_amp": True,
    "max_grad_norm": 1.0,
    "val_select_ratio": 0.10,
    "val_tune_ratio": 0.10,
    "threshold_metric": "balanced_acc",
    "default_threshold": 0.5,
    "threshold_search": {
        "start": 0.10,
        "end": 0.90,
        "step": 0.02,
    },
    "data_balance": {
        "use_weighted_sampler": False,
    },
    "augment": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.0,
        "color_jitter": 0.10,
        "random_erasing": 0.15,
    },
    "diagnostics": {
        "eval_test_each_epoch": False,
        "save_top_mistakes": 50,
    },
}

SWIN_CONFIG = deepcopy(BASE_CONFIG)
SWIN_CONFIG.update({
    "run_name": "swin_tiny_deepdetect_v3",
    "checkpoint_dir": str((MODEL_ROOT / "swin" / "deepdetect_v3" / "checkpoints").resolve()),
    "output_dir": str((OUTPUT_ROOT / "swin_deepdetect_v3").resolve()),
    "log_dir": str((OUTPUT_ROOT / "swin_deepdetect_v3" / "logs").resolve()),
    "train_log_name": "train.log",
    "test_log_name": "test.log",
    "model": {
        "arch": "swin_tiny_patch4_window7_224",
        "img_size": 224,
        "dropout": 0.20,
        "drop_path_rate": 0.15,
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
    "run_name": "efficientnet_b3_deepdetect_v3",
    "checkpoint_dir": str((MODEL_ROOT / "efficientnet_b3" / "deepdetect_v3" / "checkpoints").resolve()),
    "output_dir": str((OUTPUT_ROOT / "efficientnet_b3_deepdetect_v3").resolve()),
    "log_dir": str((OUTPUT_ROOT / "efficientnet_b3_deepdetect_v3" / "logs").resolve()),
    "train_log_name": "train.log",
    "test_log_name": "test.log",
    "model": {
        "arch": "efficientnet_b3",
        "img_size": 300,
        "dropout": 0.20,
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
    "run_name": "swin_effb3_posthoc_deepdetect_v3",
    "output_dir": str((OUTPUT_ROOT / "ensemble_posthoc_deepdetect_v3").resolve()),
    "log_dir": str((OUTPUT_ROOT / "ensemble_posthoc_deepdetect_v3" / "logs").resolve()),
    "search_on": "logits",
    "weight_search": {
        "start": 0.0,
        "end": 1.0,
        "step": 0.05,
    },
    "save_test_outputs": True,
    "swin_checkpoint": str((Path(SWIN_CONFIG["checkpoint_dir"]) / "best.pt").resolve()),
    "efficientnet_checkpoint": str((Path(EFFICIENTNET_CONFIG["checkpoint_dir"]) / "best.pt").resolve()),
})
