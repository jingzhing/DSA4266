"""Deterministic, framework-agnostic augmentation utilities.

This module replaces the previous TensorFlow runtime augmentation path with
disk-level image augmentation so both Swin (PyTorch) and EfficientNet
(TensorFlow) can consume the same prepared dataset outputs.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np

from pipeline.augmentation import AugmentationConfig, apply_transforms, augment_file_to_path
from pipeline.common import list_images

DEFAULT_PROBABILITIES = {
    "hflip": 0.5,
    "vflip": 0.1,
    "gaussian_blur": 0.25,
    "random_erase": 0.2,
    "rotate": 0.2,
    "brightness_contrast": 0.3,
    "gaussian_noise": 0.15,
    "jpeg_compression": 0.15,
}


def build_default_config() -> AugmentationConfig:
    return AugmentationConfig(
        probabilities=dict(DEFAULT_PROBABILITIES),
        erase_area_range=(0.05, 0.2),
        blur_kernel=5,
        blur_sigma_min=0.3,
        blur_sigma_max=1.5,
        rotate_degrees=12.0,
        brightness_limit=0.15,
        contrast_limit=0.2,
        noise_sigma_min=3.0,
        noise_sigma_max=12.0,
        jpeg_quality_min=45,
        jpeg_quality_max=95,
    )


def get_augmented_image(image: np.ndarray, seed: int = 4266) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return apply_transforms(image=image, cfg=build_default_config(), rng=rng)


def augment_directory(
    input_dir: str,
    output_dir: str,
    percentage: float = 0.2,
    seed: int = 4266,
) -> Dict[str, int]:
    src_root = Path(input_dir)
    dst_root = Path(output_dir)
    images = list_images(src_root)
    rng = np.random.default_rng(seed)
    n_to_generate = int(np.floor(len(images) * percentage))
    n_to_generate = max(0, min(len(images), n_to_generate))

    cfg = build_default_config()
    success = 0
    for idx in range(n_to_generate):
        src = images[idx]
        rel_parent = src.parent.relative_to(src_root)
        out_path = dst_root / rel_parent / f"{src.stem}_aug_{idx:06d}{src.suffix.lower()}"
        if augment_file_to_path(src, out_path, cfg=cfg, rng=rng):
            success += 1

    return {
        "source_images": len(images),
        "requested_augmented": n_to_generate,
        "written_augmented": success,
    }


def get_augmented_ds(input_dir: str, output_dir: str, percentage: float) -> Dict[str, int]:
    return augment_directory(input_dir=input_dir, output_dir=output_dir, percentage=percentage)


def get_new_ds(input_dir: str, output_dir: str, percentage: float) -> Dict[str, int]:
    return augment_directory(input_dir=input_dir, output_dir=output_dir, percentage=percentage)


def load_data(path: str) -> List[str]:
    return [str(p) for p in list_images(path)]
