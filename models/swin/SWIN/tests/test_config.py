from __future__ import annotations

from pathlib import Path

from pipeline.config import load_config


def test_config_sections_present() -> None:
    cfg = load_config(Path("configs/pipeline.yaml"))
    for section in [
        "project",
        "paths",
        "data",
        "audit",
        "video",
        "prepare",
        "models",
        "training",
        "evaluation",
        "inference",
        "artifacts",
    ]:
        assert section in cfg


def test_class_contract_defaults() -> None:
    cfg = load_config(Path("configs/pipeline.yaml"))
    assert cfg["data"]["class_names"] == ["real", "fake"]
