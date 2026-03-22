from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "DSA4266-ai-image-detection",
        "seed": 42,
    },
    "paths": {
        "data_root": "data",
        "raw_root": "data/raw",
        "prepared_root": "data/prepared",
        "outputs_root": "outputs/runs",
    },
    "data": {
        "dataset_id": "ayushmandatta1/deepdetect-2025",
        "dataset_version": "deepdetect-2025",
        "raw_train_subdir": "ddata/train",
        "raw_test_subdir": "ddata/test",
        "class_names": ["real", "fake"],
        "skip_download": False,
    },
    "video": {
        "enabled": False,
        "output_subdir": "video_frames",
        "blur_threshold": 7.0,
        "min_frame_stride": 100,
        "max_frame_stride": 600,
        "seed": 42,
        "cleanup_video_file": True,
    },
    "prepare": {
        "val_ratio": 0.1,
        "overwrite": False,
        "copy_test_from_raw": True,
        "augmentation": {
            "enabled": True,
            "target_class": "fake",
            "max_multiplier": 1.25,
            "probabilities": {
                "hflip": 0.5,
                "vflip": 0.1,
                "gaussian_blur": 0.25,
                "random_erase": 0.2,
            },
            "erase_area_range": [0.05, 0.2],
            "blur_kernel": 5,
            "blur_sigma_min": 0.3,
            "blur_sigma_max": 1.5,
        },
    },
    "models": {
        "swin": {
            "model_name": "swin_tiny_patch4_window7_224",
            "img_size": 224,
            "batch_size": 16,
            "epochs": 12,
            "lr": 1e-4,
            "num_workers": 2,
        },
        "efficientnet": {
            "variant": "B0",
            "img_size": 224,
            "batch_size": 16,
            "epochs": 5,
            "lr": 1e-4,
            "freeze_backbone": True,
            "dropout": 0.3,
        },
    },
    "training": {
        "threshold_metric": "balanced_acc",
        "default_threshold": 0.5,
        "runtime_augmentation": False,
    },
    "evaluation": {
        "save_confusion_matrix": True,
    },
    "inference": {
        "default_input": "",
    },
    "artifacts": {
        "tag": "default",
        "overwrite": False,
    },
}


def _deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _require_sections(cfg: Dict[str, Any]) -> None:
    required = [
        "project",
        "paths",
        "data",
        "video",
        "prepare",
        "models",
        "training",
        "evaluation",
        "inference",
        "artifacts",
    ]
    missing = [section for section in required if section not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    merged = _deep_update(copy.deepcopy(DEFAULT_CONFIG), raw_cfg)
    repo_root = cfg_path.resolve().parent.parent
    merged["_meta"] = {
        "config_path": str(cfg_path.resolve()),
        "repo_root": str(repo_root),
    }
    _require_sections(merged)
    return merged


def dump_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_path(cfg: Dict[str, Any], raw_path: str) -> Path:
    return (Path(cfg["_meta"]["repo_root"]) / raw_path).resolve()
