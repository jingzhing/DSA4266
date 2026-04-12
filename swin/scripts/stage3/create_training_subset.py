from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.glob("*") if p.is_file() and p.suffix.lower() in exts)


def _safe_link_or_copy(src: Path, dst: Path, copy_only: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_only:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _sample_deterministic(paths: List[Path], count: int, rng: random.Random) -> List[Path]:
    if count >= len(paths):
        return list(paths)
    idxs = list(range(len(paths)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:count])
    return [paths[i] for i in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic prepared subset for Phase-3 SWIN runs.")
    parser.add_argument("--prepared-root", default="data/prepared")
    parser.add_argument("--source-version", default="deepdetect-2025")
    parser.add_argument("--target-version", default="deepdetect-2025-swin-phase3-cpu")
    parser.add_argument("--train-per-class", type=int, default=1500)
    parser.add_argument("--val-per-class", type=int, default=300)
    parser.add_argument("--test-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--copy-only", action="store_true", help="Copy files instead of creating hard links")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root).resolve()
    src_root = prepared_root / args.source_version
    dst_root = prepared_root / args.target_version

    if not src_root.exists():
        raise FileNotFoundError(f"Source prepared dataset not found: {src_root}")

    if dst_root.exists():
        if not args.force:
            raise RuntimeError(f"Target exists: {dst_root}. Use --force to overwrite.")
        shutil.rmtree(dst_root)

    rng = random.Random(args.seed)
    class_names = ["real", "fake"]
    requested = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }

    summary: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        for class_name in class_names:
            src_dir = src_root / split / class_name
            dst_dir = dst_root / split / class_name
            src_images = _list_images(src_dir)
            sampled = _sample_deterministic(src_images, requested[split], rng)
            for i, src in enumerate(sampled):
                dst = dst_dir / f"{class_name}_{split}_{i:06d}{src.suffix.lower()}"
                _safe_link_or_copy(src, dst, copy_only=args.copy_only)
            summary[split][class_name] = len(sampled)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_prepared_dataset": str(src_root),
        "target_prepared_dataset": str(dst_root),
        "seed": args.seed,
        "requested_per_class": requested,
        "actual_counts": summary,
        "copy_only": bool(args.copy_only),
    }
    (dst_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
