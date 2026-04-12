from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pipeline.common import list_images


def _spec_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_dependencies(model: str) -> Dict[str, Any]:
    required: List[str] = ["numpy", "sklearn", "yaml", "PIL"]
    if model == "swin":
        required.extend(["torch", "torchvision", "timm"])
    if model == "efficientnet":
        required.extend(["tensorflow"])
    if model == "setup":
        required.extend(["kagglehub"])
    if model == "video":
        required.extend(["cv2", "yt_dlp"])
    if model == "augmentation":
        required.extend(["cv2"])

    missing = [name for name in required if not _spec_exists(name)]
    return {
        "check": "dependencies",
        "ok": len(missing) == 0,
        "missing": missing,
    }


def check_split_ratio(val_ratio: float) -> Dict[str, Any]:
    ok = 0.0 < val_ratio < 1.0
    return {
        "check": "val_ratio",
        "ok": ok,
        "value": val_ratio,
        "expected": "0 < val_ratio < 1",
    }


def check_class_folders(root_dir: Path, class_names: Iterable[str]) -> Dict[str, Any]:
    missing = [name for name in class_names if not (root_dir / name).exists()]
    return {
        "check": "class_folders",
        "ok": len(missing) == 0,
        "root": str(root_dir),
        "missing": missing,
    }


def check_non_empty_split(root_dir: Path, split: str, class_names: Iterable[str]) -> Dict[str, Any]:
    split_dir = root_dir / split
    counts = {}
    for class_name in class_names:
        counts[class_name] = len(list_images(split_dir / class_name))
    ok = all(count > 0 for count in counts.values())
    return {
        "check": f"{split}_non_empty",
        "ok": ok,
        "split_dir": str(split_dir),
        "counts": counts,
    }


def check_checkpoint_collision(checkpoint_path: Path, overwrite: bool) -> Dict[str, Any]:
    exists = checkpoint_path.exists()
    ok = overwrite or not exists
    return {
        "check": "checkpoint_collision",
        "ok": ok,
        "checkpoint_path": str(checkpoint_path),
        "exists": exists,
        "overwrite": overwrite,
    }


def summarize_preflight(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = all(check.get("ok", False) for check in checks)
    return {
        "ok": ok,
        "checks": checks,
    }
