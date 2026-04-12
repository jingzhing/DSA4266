CONFIG = {
    "train_dir": "data/deepdetect-2025/ddata/train",
    "test_dir": "data/deepdetect-2025/ddata/test",
    "val_ratio": 0.1,

    "batch_size": 16,
    "epochs": 12,
    "num_workers": 2,
    "seed": 42,

    "checkpoint_dir": "models/ensemble/v3/checkpoints",
    "output_dir": "outputs/ensemble_v3",
    "log_dir": "outputs/ensemble_v3/logs",
    "train_log_name": "ensemble_train.log",
    "test_log_name": "ensemble_test.log",

    "threshold_metric": "balanced_acc",
    "default_threshold": 0.5,
    "save_val_outputs": True,

    "ensemble": {
        "base_method": "probability_average",
        "initial_weights": {"swin": 0.5, "efficientnet": 0.5},
        "learn_weights_post_training": True,
        "weight_search_start": 0.0,
        "weight_search_end": 1.0,
        "weight_search_step": 0.02,
        "top_k_to_print": 10,
    },

    "efficientnet": {
        "model_name": "efficientnet_b2",
        "img_size": 260,
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "freeze_backbone": False,
        "label_smoothing": 0.0,
    },

    "swin": {
        "model_name": "swin_tiny_patch4_window7_224",
        "img_size": 224,
        "dropout": 0.2,
        "drop_path_rate": 0.2,
        "lr": 7e-5,
        "weight_decay": 5e-5,
        "freeze_backbone": False,
        "label_smoothing": 0.05,
    },

    "augment": {
        "horizontal_flip": 0.5,
        "color_jitter": 0.15,
        "random_erasing": 0.1,
    },
}
