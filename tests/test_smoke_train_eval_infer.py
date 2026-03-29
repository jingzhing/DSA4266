from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pipeline.config import load_config
from pipeline.stages import run_eval, run_infer, run_prepare, run_setup, run_train
from tests.helpers import create_tiny_raw_dataset


def _cfg_for_training(tmp_path: Path) -> dict:
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
    cfg["models"] = {
        "swin": dict(cfg["models"]["swin"]),
        "efficientnet": dict(cfg["models"]["efficientnet"]),
    }
    cfg["models"]["swin"]["epochs"] = 1
    cfg["models"]["swin"]["batch_size"] = 2
    cfg["models"]["swin"]["num_workers"] = 0
    cfg["models"]["efficientnet"]["epochs"] = 1
    cfg["models"]["efficientnet"]["batch_size"] = 2
    cfg["models"]["efficientnet"]["img_size"] = 128
    cfg["artifacts"] = dict(cfg["artifacts"])
    cfg["artifacts"]["tag"] = "smoke"
    cfg["_meta"] = dict(cfg["_meta"])
    cfg["_meta"]["repo_root"] = str(tmp_path.resolve())
    return cfg


def _prepare_fixture(cfg: dict, tmp_path: Path) -> Path:
    raw_dataset_root = tmp_path / "data" / "raw" / "deepdetect-2025"
    create_tiny_raw_dataset(raw_dataset_root)
    run_setup(cfg, force=False)
    run_prepare(cfg, with_video=False, video_urls=None, force=True)
    return Path(cfg["paths"]["prepared_root"]).resolve() / cfg["data"]["dataset_version"] / "test"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
@pytest.mark.skipif(importlib.util.find_spec("timm") is None, reason="timm not installed")
def test_swin_train_eval_infer_smoke(tmp_path: Path) -> None:
    cfg = _cfg_for_training(tmp_path)
    infer_input = _prepare_fixture(cfg, tmp_path)

    train_result = run_train(cfg, model_name="swin")
    run_dir = Path(train_result["run_dir"])
    eval_result = run_eval(cfg, model_name="swin", run_dir=run_dir)
    infer_result = run_infer(cfg, model_name="swin", input_dir=infer_input, run_dir=run_dir)

    assert run_dir.exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()
    assert eval_result["metrics"]["accuracy"] >= 0.0
    assert infer_result["prediction_count"] > 0


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not installed")
def test_efficientnet_train_eval_infer_smoke(tmp_path: Path) -> None:
    cfg = _cfg_for_training(tmp_path)
    infer_input = _prepare_fixture(cfg, tmp_path)

    train_result = run_train(cfg, model_name="efficientnet")
    run_dir = Path(train_result["run_dir"])
    eval_result = run_eval(cfg, model_name="efficientnet", run_dir=run_dir)
    infer_result = run_infer(cfg, model_name="efficientnet", input_dir=infer_input, run_dir=run_dir)

    assert run_dir.exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()
    assert eval_result["metrics"]["accuracy"] >= 0.0
    assert infer_result["prediction_count"] > 0
