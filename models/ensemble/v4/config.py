from copy import deepcopy

BASE_CONFIG = {
    "train_dir": "data/deepdetect-2025/ddata/train",
    "test_dir": "data/deepdetect-2025/ddata/test",
    "val_select_ratio": 0.10,
    "val_tune_ratio": 0.10,
    "split_path": "outputs/deepdetect/fixed_split_seed42.json",
    "seed": 42,
    "batch_size": 16,
    "num_workers": 2,
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
        "color_jitter": 0.15,
        "random_erasing": 0.0,
    },
    "diagnostics": {
        "eval_test_each_epoch": False,
        "save_top_mistakes": 50,
    },
}

SWIN_CONFIG = deepcopy(BASE_CONFIG)
SWIN_CONFIG.update({
    "run_name": "swin_tiny_v2_deepdetect",
    "checkpoint_dir": "models/swin/v2_deepdetect/checkpoints",
    "output_dir": "outputs/swin_v2_deepdetect",
    "log_dir": "outputs/swin_v2_deepdetect/logs",
    "train_log_name": "train.log",
    "test_log_name": "test.log",
    "model": {
        "arch": "swin_tiny_patch4_window7_224",
        "img_size": 224,
        "dropout": 0.2,
        "drop_path_rate": 0.2,
        "freeze_backbone": False,
    },
    "train": {
        "epochs": 1,
        "lr": 1e-4,
        "weight_decay": 5e-5,
        "label_smoothing": 0.05,
        "optimizer": "adamw",
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 5e-4,
        "scheduler": None,
    },
})

EFFICIENTNET_CONFIG = deepcopy(BASE_CONFIG)
EFFICIENTNET_CONFIG.update({
    "run_name": "efficientnet_b2_v2_deepdetect",
    "checkpoint_dir": "models/efficientnet_b2/v2_deepdetect/checkpoints",
    "output_dir": "outputs/efficientnet_b2_v2_deepdetect",
    "log_dir": "outputs/efficientnet_b2_v2_deepdetect/logs",
    "train_log_name": "train.log",
    "test_log_name": "test.log",
    "model": {
        "arch": "efficientnet_b2",
        "img_size": 260,
        "dropout": 0.3,
        "drop_path_rate": 0.0,
        "freeze_backbone": False,
        "partial_finetune": True,
        "partial_finetune_blocks": 30,
    },
    "train": {
        "epochs": 1,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "optimizer": "adamw",
        "early_stopping_patience": 3,
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
    "run_name": "swin_effb2_posthoc_deepdetect_v1",
    "output_dir": "outputs/ensemble_posthoc_deepdetect_v1",
    "log_dir": "outputs/ensemble_posthoc_deepdetect_v1/logs",
    "search_on": "logits",
    "weight_search": {
        "start": 0.0,
        "end": 1.0,
        "step": 0.05,
    },
    "save_test_outputs": True,
    "swin_checkpoint": "models/swin/v2_deepdetect/checkpoints/best.pt",
    "efficientnet_checkpoint": "models/efficientnet_b2/v2_deepdetect/checkpoints/best.pt",
})
