import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def copy_files(files, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for src in files:
        dst = destination / src.name
        if dst.exists():
            stem = src.stem
            suffix = src.suffix
            i = 1
            while True:
                candidate = destination / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1
        shutil.copy2(src, dst)


def make_balanced_train_dataset(
    source_train_dir="data/video_frames_split/train",
    output_train_dir="data/video_frames_balanced/train",
    fake_label="fake",
    real_label="real",
    seed=42,
):
    source_train_dir = Path(source_train_dir)
    output_train_dir = Path(output_train_dir)

    fake_src = source_train_dir / fake_label
    real_src = source_train_dir / real_label

    fake_files = collect_images(fake_src)
    real_files = collect_images(real_src)

    print(f"Found fake images: {len(fake_files)}")
    print(f"Found real images: {len(real_files)}")

    if len(fake_files) == 0 or len(real_files) == 0:
        raise RuntimeError("One of the classes is empty. Check your train folder.")

    target_count = min(len(fake_files), len(real_files))
    rng = random.Random(seed)

    fake_selected = rng.sample(fake_files, target_count) if len(fake_files) > target_count else fake_files
    real_selected = rng.sample(real_files, target_count) if len(real_files) > target_count else real_files

    if output_train_dir.exists():
        shutil.rmtree(output_train_dir)

    fake_out = output_train_dir / fake_label
    real_out = output_train_dir / real_label

    copy_files(fake_selected, fake_out)
    copy_files(real_selected, real_out)

    print("\nBalanced train dataset created.")
    print(f"Output folder: {output_train_dir}")
    print(f"Fake kept: {len(fake_selected)}")
    print(f"Real kept: {len(real_selected)}")
    print(f"Total train images: {len(fake_selected) + len(real_selected)}")


if __name__ == "__main__":
    make_balanced_train_dataset()
