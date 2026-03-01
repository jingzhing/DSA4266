CONFIG = {
    "train_dir": "data/deepdetect-2025_dddata/train",
    "test_dir": "data/deepdetect-2025_dddata/test",
    "val_ratio": 0.1,

    "img_size": 224,
    "batch_size": 16,

    "epochs": 12,
    "lr": 1e-4,

    "num_workers": 2,
    "model_name": "swin_tiny_patch4_window7_224",
    "seed": 42,

    "checkpoint_dir": "models/swin/v1/checkpoints",
    "output_dir": "outputs",

    "threshold_metric": "balanced_acc",
    "default_threshold": 0.5,
}