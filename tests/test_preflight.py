from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import load_config
from pipeline.stages import run_prepare
from tests.helpers import create_tiny_raw_dataset


def _cfg_for_tmp(tmp_path: Path) -> dict:
    cfg = load_config("configs/pipeline.yaml")
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["raw_root"] = str((tmp_path / "data" / "raw").as_posix())
    cfg["paths"]["prepared_root"] = str((tmp_path / "data" / "prepared").as_posix())
    cfg["paths"]["outputs_root"] = str((tmp_path / "outputs" / "runs").as_posix())
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["skip_download"] = True
    cfg["prepare"] = dict(cfg["prepare"])
    cfg["prepare"]["overwrite"] = True
    cfg["prepare"]["augmentation"] = dict(cfg["prepare"]["augmentation"])
    cfg["prepare"]["augmentation"]["enabled"] = False
    cfg["_meta"] = dict(cfg["_meta"])
    cfg["_meta"]["repo_root"] = str(tmp_path.resolve())
    return cfg


def test_invalid_val_ratio_fails(tmp_path: Path) -> None:
    cfg = _cfg_for_tmp(tmp_path)
    cfg["prepare"]["val_ratio"] = 1.5
    create_tiny_raw_dataset(tmp_path / "data" / "raw" / "deepdetect-2025")
    with pytest.raises(RuntimeError):
        run_prepare(cfg, with_video=False, video_urls=None, force=True)


def test_missing_class_folder_fails(tmp_path: Path) -> None:
    cfg = _cfg_for_tmp(tmp_path)
    raw_root = tmp_path / "data" / "raw" / "deepdetect-2025"
    create_tiny_raw_dataset(raw_root)
    # Remove one required class directory.
    missing_dir = raw_root / "ddata" / "train" / "fake"
    for file in missing_dir.glob("*"):
        file.unlink()
    missing_dir.rmdir()

    with pytest.raises(RuntimeError):
        run_prepare(cfg, with_video=False, video_urls=None, force=True)
