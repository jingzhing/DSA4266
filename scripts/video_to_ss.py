"""Compatibility wrappers for optional video frame enrichment stage."""

from pathlib import Path
from typing import Any, Dict

from pipeline.video import download_and_extract, download_video as _download_video, extract_clear_frames


def download_video(url: str, download_dir: str = "data/raw/video_frames/downloads") -> str:
    path = _download_video(url=url, download_dir=Path(download_dir))
    return str(path)


def get_vid_frames(
    path: str,
    output_dir: str = "data/raw/video_frames/extracted",
    blur_threshold: float = 7.0,
    min_frame_stride: int = 100,
    max_frame_stride: int = 600,
    seed: int = 42,
) -> Dict[str, Any]:
    return extract_clear_frames(
        video_path=Path(path),
        output_dir=Path(output_dir),
        blur_threshold=blur_threshold,
        min_frame_stride=min_frame_stride,
        max_frame_stride=max_frame_stride,
        seed=seed,
    )


def download_and_process(
    url: str,
    output_dir: str = "data/raw/video_frames/extracted",
    download_dir: str = "data/raw/video_frames/downloads",
    cleanup_video_file: bool = True,
) -> Dict[str, Any]:
    return download_and_extract(
        url=url,
        download_dir=Path(download_dir),
        output_dir=Path(output_dir),
        blur_threshold=7.0,
        min_frame_stride=100,
        max_frame_stride=600,
        seed=42,
        cleanup_video_file=cleanup_video_file,
    )

