from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def create_tiny_raw_dataset(raw_dataset_root: Path) -> None:
    for split in ["train", "test"]:
        for class_name in ["real", "fake"]:
            for idx in range(4):
                make_image(
                    raw_dataset_root / "ddata" / split / class_name / f"{class_name}_{idx}.jpg",
                    seed=(idx + (0 if class_name == "real" else 100)),
                )

