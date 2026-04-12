"""Compatibility entrypoint for dataset setup.

This script now delegates to the unified pipeline stage:
    python -m pipeline.cli setup --config configs/pipeline.yaml
"""

from pipeline.config import load_config
from pipeline.stages import run_setup
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    result = run_setup(cfg=cfg, force=False)
    print(result)


if __name__ == "__main__":
    main()
