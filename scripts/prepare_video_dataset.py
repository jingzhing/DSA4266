import random
import shutil
from pathlib import Path
import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def collect_videos(folder):
    folder = Path(folder)
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS])


def split_videos(video_paths, train_ratio=0.8, test_ratio=0.2, seed=42):
    if abs(train_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + test_ratio must sum to 1")

    video_paths = list(video_paths)
    rng = random.Random(seed)
    rng.shuffle(video_paths)

    n = len(video_paths)
    n_train = int(n * train_ratio)

    train_videos = video_paths[:n_train]
    test_videos = video_paths[n_train:]

    return train_videos, test_videos


def extract_frames_from_video(video_path, output_dir, seconds_interval=5, max_frames=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    frame_interval = max(1, int(round(fps * seconds_interval)))

    saved = 0
    frame_idx = 0
    stem = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_name = f"{stem}_frame_{frame_idx:06d}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def process_split(video_paths, output_dir, seconds_interval=5, max_frames=None):
    total = 0
    for video_path in video_paths:
        saved = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            seconds_interval=seconds_interval,
            max_frames=max_frames,
        )
        print(f"{video_path.name}: saved {saved} frames")
        total += saved
    return total


def clear_output_folder(folder):
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def main():
    manipulated_dir = Path.home() / "Downloads" / "DFD_manipulated_sequences"
    original_dir = Path.home() / "Downloads" / "DFD_original_sequences"

    output_root = Path("data/video_frames_split")

    train_ratio = 0.8
    test_ratio = 0.2
    seed = 42
    seconds_interval = 5
    max_frames_per_video = None

    manipulated_videos = collect_videos(manipulated_dir)
    original_videos = collect_videos(original_dir)

    print(f"Manipulated videos found: {len(manipulated_videos)}")
    print(f"Original videos found: {len(original_videos)}")

    if len(manipulated_videos) == 0 or len(original_videos) == 0:
        raise RuntimeError("No videos found. Check your Downloads folder paths.")

    m_train, m_test = split_videos(
        manipulated_videos,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    o_train, o_test = split_videos(
        original_videos,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    clear_output_folder(output_root)

    paths = {
        "train_fake": output_root / "train" / "fake",
        "train_real": output_root / "train" / "real",
        "test_fake": output_root / "test" / "fake",
        "test_real": output_root / "test" / "real",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    print("\nExtracting manipulated train videos...")
    train_fake_count = process_split(m_train, paths["train_fake"], seconds_interval, max_frames_per_video)

    print("\nExtracting manipulated test videos...")
    test_fake_count = process_split(m_test, paths["test_fake"], seconds_interval, max_frames_per_video)

    print("\nExtracting original train videos...")
    train_real_count = process_split(o_train, paths["train_real"], seconds_interval, max_frames_per_video)

    print("\nExtracting original test videos...")
    test_real_count = process_split(o_test, paths["test_real"], seconds_interval, max_frames_per_video)

    print("\nDone.")
    print(f"Train fake frames: {train_fake_count}")
    print(f"Train real frames: {train_real_count}")
    print(f"Test fake frames:  {test_fake_count}")
    print(f"Test real frames:  {test_real_count}")


if __name__ == "__main__":
    main()