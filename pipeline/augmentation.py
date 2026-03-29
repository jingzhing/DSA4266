from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class AugmentationConfig:
    probabilities: Dict[str, float]
    erase_area_range: Tuple[float, float]
    blur_kernel: int
    blur_sigma_min: float
    blur_sigma_max: float
    rotate_degrees: float
    brightness_limit: float
    contrast_limit: float
    noise_sigma_min: float
    noise_sigma_max: float
    jpeg_quality_min: int
    jpeg_quality_max: int


def _apply_hflip(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.flip(image, 1)


def _apply_vflip(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.flip(image, 0)


def _apply_blur(
    image: np.ndarray,
    rng: np.random.Generator,
    kernel_size: int,
    sigma_min: float,
    sigma_max: float,
) -> np.ndarray:
    import cv2

    sigma = float(rng.uniform(sigma_min, sigma_max))
    k = max(3, kernel_size)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), sigma)


def _apply_random_erase(
    image: np.ndarray,
    rng: np.random.Generator,
    erase_area_range: Tuple[float, float],
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    area = h * w
    erase_ratio = float(rng.uniform(erase_area_range[0], erase_area_range[1]))
    erase_area = max(1, int(area * erase_ratio))

    erase_h = max(1, int(math.sqrt(erase_area)))
    erase_w = max(1, int(erase_area / erase_h))
    erase_h = min(erase_h, h)
    erase_w = min(erase_w, w)

    top = int(rng.integers(0, max(1, h - erase_h + 1)))
    left = int(rng.integers(0, max(1, w - erase_w + 1)))
    out[top : top + erase_h, left : left + erase_w] = 0
    return out


def _apply_rotate(
    image: np.ndarray,
    rng: np.random.Generator,
    rotate_degrees: float,
) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    angle = float(rng.uniform(-rotate_degrees, rotate_degrees))
    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _apply_brightness_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness_limit: float,
    contrast_limit: float,
) -> np.ndarray:
    alpha = float(1.0 + rng.uniform(-contrast_limit, contrast_limit))
    beta = float(rng.uniform(-brightness_limit, brightness_limit) * 255.0)
    out = image.astype(np.float32) * alpha + beta
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _apply_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    noise_sigma_min: float,
    noise_sigma_max: float,
) -> np.ndarray:
    sigma = float(rng.uniform(noise_sigma_min, noise_sigma_max))
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _apply_jpeg_compression(
    image: np.ndarray,
    rng: np.random.Generator,
    jpeg_quality_min: int,
    jpeg_quality_max: int,
) -> np.ndarray:
    import cv2

    quality = int(rng.integers(jpeg_quality_min, jpeg_quality_max + 1))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return image
    return decoded


def apply_transforms(
    image: np.ndarray,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    out = image.copy()
    transforms_applied: List[str] = []

    if rng.random() < cfg.probabilities.get("hflip", 0.0):
        out = _apply_hflip(out)
        transforms_applied.append("hflip")
    if rng.random() < cfg.probabilities.get("vflip", 0.0):
        out = _apply_vflip(out)
        transforms_applied.append("vflip")
    if rng.random() < cfg.probabilities.get("gaussian_blur", 0.0):
        out = _apply_blur(out, rng, cfg.blur_kernel, cfg.blur_sigma_min, cfg.blur_sigma_max)
        transforms_applied.append("gaussian_blur")
    if rng.random() < cfg.probabilities.get("random_erase", 0.0):
        out = _apply_random_erase(out, rng, cfg.erase_area_range)
        transforms_applied.append("random_erase")
    if rng.random() < cfg.probabilities.get("rotate", 0.0):
        out = _apply_rotate(out, rng, cfg.rotate_degrees)
        transforms_applied.append("rotate")
    if rng.random() < cfg.probabilities.get("brightness_contrast", 0.0):
        out = _apply_brightness_contrast(out, rng, cfg.brightness_limit, cfg.contrast_limit)
        transforms_applied.append("brightness_contrast")
    if rng.random() < cfg.probabilities.get("gaussian_noise", 0.0):
        out = _apply_gaussian_noise(out, rng, cfg.noise_sigma_min, cfg.noise_sigma_max)
        transforms_applied.append("gaussian_noise")
    if rng.random() < cfg.probabilities.get("jpeg_compression", 0.0):
        out = _apply_jpeg_compression(out, rng, cfg.jpeg_quality_min, cfg.jpeg_quality_max)
        transforms_applied.append("jpeg_compression")

    # Ensure at least one augmentation is applied for deterministic data expansion.
    if not transforms_applied:
        out = _apply_hflip(out)
    return out


def augment_file_to_path(
    src_path: Path,
    dst_path: Path,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
) -> bool:
    import cv2

    image = cv2.imread(str(src_path))
    if image is None:
        return False
    augmented = apply_transforms(image, cfg, rng)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst_path), augmented))
