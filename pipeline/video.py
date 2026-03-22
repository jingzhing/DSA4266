from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def download_video(url: str, download_dir: Path) -> Path:
    import yt_dlp

    download_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(download_dir / "yt_dl_%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        requested = info.get("requested_downloads") or []
        if requested:
            path = requested[0].get("filepath")
            if path:
                return Path(path).resolve()
        fallback = Path(ydl.prepare_filename(info))
        if fallback.suffix.lower() != ".mp4":
            mp4_candidate = fallback.with_suffix(".mp4")
            if mp4_candidate.exists():
                fallback = mp4_candidate
        return fallback.resolve()


def extract_clear_frames(
    video_path: Path,
    output_dir: Path,
    blur_threshold: float,
    min_frame_stride: int,
    max_frame_stride: int,
    seed: int,
) -> Dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    saved = 0
    failures = 0

    while frame_index < total_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if ok:
            laplace_var = float(cv2.Laplacian(frame, cv2.CV_64F).var())
            if laplace_var >= blur_threshold:
                out_file = output_dir / f"{video_path.stem}_f{frame_index:07d}.jpg"
                if cv2.imwrite(str(out_file), frame):
                    saved += 1
                else:
                    failures += 1
        step = int(rng.integers(min_frame_stride, max_frame_stride + 1))
        frame_index += max(1, step)

    capture.release()
    return {
        "video_path": str(video_path),
        "total_frames": total_frames,
        "saved_frames": saved,
        "write_failures": failures,
        "blur_threshold": blur_threshold,
        "min_frame_stride": min_frame_stride,
        "max_frame_stride": max_frame_stride,
        "seed": seed,
    }


def download_and_extract(
    url: str,
    download_dir: Path,
    output_dir: Path,
    blur_threshold: float,
    min_frame_stride: int,
    max_frame_stride: int,
    seed: int,
    cleanup_video_file: bool,
) -> Dict[str, Any]:
    video_path = download_video(url, download_dir=download_dir)
    stats = extract_clear_frames(
        video_path=video_path,
        output_dir=output_dir,
        blur_threshold=blur_threshold,
        min_frame_stride=min_frame_stride,
        max_frame_stride=max_frame_stride,
        seed=seed,
    )
    if cleanup_video_file and video_path.exists():
        os.remove(video_path)
    return stats

