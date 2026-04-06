from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pipeline.config import load_config
from pipeline.preflight import check_imagefolder_class_index_mapping
from pipeline.stages import run_prepare
from tests.helpers import create_tiny_raw_dataset, make_image


def _cfg_for_tmp(tmp_path: Path) -> dict:
    cfg = load_config("configs/pipeline.yaml")
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["raw_root"] = str((tmp_path / "data" / "raw").as_posix())
    cfg["paths"]["prepared_root"] = str((tmp_path / "data" / "prepared").as_posix())
    cfg["paths"]["outputs_root"] = str((tmp_path / "outputs" / "runs").as_posix())
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["skip_download"] = True
    cfg["video"] = dict(cfg["video"])
    cfg["video"]["enabled"] = False
    cfg["video"]["urls"] = []
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


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="torchvision not installed")
def test_imagefolder_class_index_mapping_reports_split_mappings(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    for split in ["train", "val", "test"]:
        make_image(prepared_root / split / "real" / f"real_{split}.jpg", seed=1)
        make_image(prepared_root / split / "fake" / f"fake_{split}.jpg", seed=2)

    check = check_imagefolder_class_index_mapping(prepared_root, ["real", "fake"])
    assert check["ok"] is True
    assert check["resolved_fake_idx"] == 0
    assert check["mapping_by_split"]["train"] == {"fake": 0, "real": 1}
    assert check["matches_config_order"] is False


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="torchvision not installed")
def test_imagefolder_class_index_mapping_fails_when_required_class_missing(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    for split in ["train", "val", "test"]:
        make_image(prepared_root / split / "real" / f"real_{split}.jpg", seed=3)

    check = check_imagefolder_class_index_mapping(prepared_root, ["real", "fake"])
    assert check["ok"] is False
