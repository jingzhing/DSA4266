from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = ("real", "fake")
EXCLUDED_DIRS = {"_quarantine"}


def _is_excluded(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return any(part in EXCLUDED_DIRS for part in rel_parts)


def list_image_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and not _is_excluded(path, root)
    )


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # nosec - integrity grouping only, not security.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_one(path: Path, root: Path) -> Dict[str, object]:
    relpath = str(path.relative_to(root)).replace("\\", "/")
    split = None
    class_name = None
    parts = [part.lower() for part in path.parts]
    if "train" in parts:
        split = "train"
    elif "test" in parts:
        split = "test"
    elif "val" in parts:
        split = "val"

    for candidate in CLASS_NAMES:
        if candidate in parts:
            class_name = candidate
            break

    result: Dict[str, object] = {
        "relpath": relpath,
        "ext": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "split": split,
        "class": class_name,
        "ok": True,
        "width": None,
        "height": None,
        "mode": None,
        "error": None,
        "md5": None,
    }

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            result["width"], result["height"] = image.size
            result["mode"] = image.mode
        result["md5"] = md5_file(path)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        result["ok"] = False
        result["error"] = str(exc)
    return result


def summarize_structure(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if _is_excluded(path, root):
            continue
        rel = "." if path == root else str(path.relative_to(root)).replace("\\", "/")
        files = sum(1 for p in path.iterdir() if p.is_file() and not _is_excluded(p, root))
        dirs = sum(1 for p in path.iterdir() if p.is_dir() and not _is_excluded(p, root))
        rows.append(
            {
                "path": rel,
                "file_count_direct": files,
                "dir_count_direct": dirs,
            }
        )
    return rows


def average(values: Iterable[int]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list)) / float(len(values_list))


def render_schema_md(inventory: Dict[str, object]) -> str:
    totals = inventory["totals"]
    class_counts = inventory["class_distribution"]
    split_counts = inventory["split_distribution"]
    duplicates = inventory["duplicates_md5_summary"]
    corrupt = inventory["corrupted_files"]
    avg_res = inventory["average_resolution"]

    lines: List[str] = []
    lines.append("# DeepDetect 2025 Dataset Schema")
    lines.append("")
    lines.append("## Source")
    lines.append(f"- Dataset ID: `{inventory['dataset_id']}`")
    lines.append(f"- Raw Root: `{inventory['raw_root']}`")
    lines.append(f"- Generated At (UTC): `{inventory['generated_at_utc']}`")
    lines.append("")
    lines.append("## Expected Contract")
    lines.append("- `ddata/train/{real,fake}`")
    lines.append("- `ddata/test/{real,fake}`")
    lines.append("")
    lines.append("## Totals")
    lines.append(f"- Total image files: **{totals['total_images']}**")
    lines.append(f"- Total bytes (images): **{totals['total_bytes_images']}**")
    lines.append(f"- Corrupted image files: **{totals['corrupted_files']}**")
    lines.append(f"- Duplicate groups (MD5): **{duplicates['duplicate_groups']}**")
    lines.append(f"- Duplicate files involved (MD5): **{duplicates['duplicate_files']}**")
    lines.append("")
    lines.append("## Distribution")
    lines.append(f"- By class: `{class_counts}`")
    lines.append(f"- By split: `{split_counts}`")
    lines.append(
        f"- Average resolution (valid images): **{avg_res['width']:.2f} x {avg_res['height']:.2f}**"
    )
    lines.append("")
    lines.append("## Integrity Notes")
    if corrupt:
        lines.append(f"- Corrupted files detected: {len(corrupt)}")
        preview = corrupt[:10]
        for row in preview:
            lines.append(f"  - `{row['relpath']}`: {row['error']}")
        if len(corrupt) > len(preview):
            lines.append(f"  - ... and {len(corrupt) - len(preview)} more")
    else:
        lines.append("- No corrupted files detected in image decode/verify pass.")
    lines.append("")
    lines.append("## Directory Structure Snapshot")
    for row in inventory["directory_structure"][:50]:
        lines.append(
            f"- `{row['path']}`: files={row['file_count_direct']}, dirs={row['dir_count_direct']}"
        )
    if len(inventory["directory_structure"]) > 50:
        lines.append(f"- ... and {len(inventory['directory_structure']) - 50} more directories")
    lines.append("")
    lines.append("## Stage-1 Gate Summary")
    lines.append("- Use `dataset_inventory.json` as machine-readable source of truth.")
    lines.append("- Confirm class balance and duplicates before Stage-2 dataset audit.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset inventory/schema for Stage-1 ingestion.")
    parser.add_argument("--dataset-root", required=True, help="Dataset folder (e.g. data/raw/deepdetect_2025)")
    parser.add_argument("--dataset-id", default="ayushmandatta1/deepdetect-2025")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-duplicate-groups", type=int, default=5000)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    image_files = list_image_files(root)
    inspections: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(inspect_one, path, root) for path in image_files]
        for future in as_completed(futures):
            inspections.append(future.result())
    inspections.sort(key=lambda row: row["relpath"])  # deterministic output

    total_bytes_images = int(sum(int(row["bytes"]) for row in inspections))
    valid_rows = [row for row in inspections if row["ok"]]
    corrupted_rows = [
        {"relpath": row["relpath"], "error": row["error"]}
        for row in inspections
        if not row["ok"]
    ]

    class_distribution: Dict[str, int] = Counter(
        str(row["class"]) if row["class"] is not None else "unknown" for row in inspections
    )
    split_distribution: Dict[str, int] = Counter(
        str(row["split"]) if row["split"] is not None else "unknown" for row in inspections
    )

    width_avg = average(int(row["width"]) for row in valid_rows if row["width"] is not None)
    height_avg = average(int(row["height"]) for row in valid_rows if row["height"] is not None)

    md5_to_paths: Dict[str, List[str]] = defaultdict(list)
    for row in valid_rows:
        md5_to_paths[str(row["md5"])].append(str(row["relpath"]))

    duplicate_groups = [
        {"md5": md5, "count": len(paths), "files": sorted(paths)}
        for md5, paths in md5_to_paths.items()
        if len(paths) > 1
    ]
    duplicate_groups.sort(key=lambda row: (-int(row["count"]), str(row["md5"])))
    duplicate_files = int(sum(int(row["count"]) for row in duplicate_groups))
    duplicate_group_count = int(len(duplicate_groups))

    truncated = False
    if len(duplicate_groups) > args.max_duplicate_groups:
        duplicate_groups = duplicate_groups[: args.max_duplicate_groups]
        truncated = True

    inventory = {
        "dataset_id": args.dataset_id,
        "raw_root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "total_images": len(inspections),
            "total_bytes_images": total_bytes_images,
            "valid_images": len(valid_rows),
            "corrupted_files": len(corrupted_rows),
        },
        "class_distribution": dict(class_distribution),
        "split_distribution": dict(split_distribution),
        "average_resolution": {
            "width": width_avg,
            "height": height_avg,
        },
        "directory_structure": summarize_structure(root),
        "corrupted_files": corrupted_rows,
        "duplicates_md5_summary": {
            "duplicate_groups": duplicate_group_count,
            "duplicate_files": duplicate_files,
            "groups_truncated": truncated,
            "max_groups_kept": args.max_duplicate_groups,
            "groups_kept": len(duplicate_groups),
        },
        "duplicates_md5_groups": duplicate_groups,
    }

    out_inventory = root / "dataset_inventory.json"
    out_schema = root / "dataset_schema.md"
    out_inventory.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    out_schema.write_text(render_schema_md(inventory), encoding="utf-8")

    print(f"Wrote {out_inventory}")
    print(f"Wrote {out_schema}")
    print(
        json.dumps(
            {
                "total_images": inventory["totals"]["total_images"],
                "corrupted_files": inventory["totals"]["corrupted_files"],
                "duplicate_groups": inventory["duplicates_md5_summary"]["duplicate_groups"],
                "duplicate_files": inventory["duplicates_md5_summary"]["duplicate_files"],
                "class_distribution": inventory["class_distribution"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
