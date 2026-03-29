from __future__ import annotations

from pathlib import Path

from pipeline.config import load_config
from pipeline.stages import run_prepare, run_setup
from tests.helpers import create_tiny_raw_dataset


def _write_test_config(base_cfg: dict, tmp_path: Path) -> dict:
    cfg = dict(base_cfg)
    cfg["paths"] = dict(base_cfg["paths"])
    cfg["paths"]["raw_root"] = str((tmp_path / "data" / "raw").as_posix())
    cfg["paths"]["prepared_root"] = str((tmp_path / "data" / "prepared").as_posix())
    cfg["paths"]["outputs_root"] = str((tmp_path / "outputs" / "runs").as_posix())
    cfg["data"] = dict(base_cfg["data"])
    cfg["data"]["skip_download"] = True
    cfg["video"] = dict(base_cfg["video"])
    cfg["video"]["enabled"] = False
    cfg["video"]["urls"] = []
    cfg["prepare"] = dict(base_cfg["prepare"])
    cfg["prepare"]["overwrite"] = True
    cfg["prepare"]["augmentation"] = dict(base_cfg["prepare"]["augmentation"])
    cfg["prepare"]["augmentation"]["enabled"] = False
    cfg["_meta"] = dict(base_cfg["_meta"])
    cfg["_meta"]["repo_root"] = str(tmp_path.resolve())
    return cfg


def test_prepare_creates_manifest_and_splits(tmp_path: Path) -> None:
    base_cfg = load_config("configs/pipeline.yaml")
    cfg = _write_test_config(base_cfg, tmp_path)

    raw_dataset_root = tmp_path / "data" / "raw" / "deepdetect-2025"
    create_tiny_raw_dataset(raw_dataset_root)
    run_setup(cfg, force=False)
    result = run_prepare(cfg, with_video=False, video_urls=None, force=True)

    prepared_root = Path(result["prepared_root"])
    manifest_path = Path(result["manifest_path"])
    assert prepared_root.exists()
    assert manifest_path.exists()
    assert (prepared_root / "train" / "real").exists()
    assert (prepared_root / "val" / "fake").exists()
    assert (prepared_root / "test" / "real").exists()
