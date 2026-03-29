from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, UnidentifiedImageError

from pipeline.common import list_images, write_json


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(relpath: str, sha256: str) -> str:
    return hashlib.sha1(f"{relpath}|{sha256}".encode("utf-8")).hexdigest()[:20]


def _iter_split_roots(raw_dataset_dir: Path, raw_train_subdir: str, raw_test_subdir: str) -> Iterable[tuple[str, Path]]:
    yield "train", raw_dataset_dir / raw_train_subdir
    yield "test", raw_dataset_dir / raw_test_subdir


def build_manifest_v1(
    *,
    raw_dataset_dir: Path,
    class_names: List[str],
    raw_train_subdir: str,
    raw_test_subdir: str,
    dataset_id: str,
    out_manifest_path: Path,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []

    for split_name, split_root in _iter_split_roots(raw_dataset_dir, raw_train_subdir, raw_test_subdir):
        for class_name in class_names:
            class_root = split_root / class_name
            for image_path in list_images(class_root):
                relpath = str(image_path.relative_to(raw_dataset_dir)).replace("\\", "/")
                sha256 = _sha256_file(image_path)
                records.append(
                    {
                        "sample_id": _sample_id(relpath, sha256),
                        "relpath": relpath,
                        "abspath": str(image_path.resolve()),
                        "sha256": sha256,
                        "bytes": int(image_path.stat().st_size),
                        "label": class_name,
                        "split": split_name,
                        "source_hint": image_path.parent.name,
                    }
                )

    records.sort(key=lambda row: row["relpath"])
    payload = {
        "schema_version": "manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "raw_dataset_dir": str(raw_dataset_dir.resolve()),
        "class_names": class_names,
        "record_count": len(records),
        "records": records,
    }
    write_json(out_manifest_path, payload)

    return {
        "manifest_path": str(out_manifest_path),
        "record_count": len(records),
    }


def run_quality_audit_v1(
    *,
    manifest_path: Path,
    out_summary_path: Path,
    out_duplicates_path: Path,
    out_assertions_path: Path,
    decode_failed_rate_threshold: float,
) -> Dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = payload["records"]

    decode_failed: List[Dict[str, Any]] = []
    widths: List[int] = []
    heights: List[int] = []
    mode_counts: Counter[str] = Counter()
    ext_by_label: Dict[str, Counter[str]] = defaultdict(Counter)
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    sha_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in records:
        path = Path(row["abspath"])
        ext = path.suffix.lower()
        split_counts[str(row["split"])] += 1
        label_counts[str(row["label"])] += 1
        ext_by_label[str(row["label"])][ext] += 1
        sha_groups[str(row["sha256"])].append(row)

        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                w, h = image.size
                mode_counts[str(image.mode)] += 1
                widths.append(int(w))
                heights.append(int(h))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            decode_failed.append({"relpath": row["relpath"], "error": str(exc)})

    duplicate_groups = []
    cross_split_duplicate_groups = 0
    cross_label_duplicate_groups = 0
    for sha256, group_rows in sha_groups.items():
        if len(group_rows) <= 1:
            continue
        splits = sorted({str(r["split"]) for r in group_rows})
        labels = sorted({str(r["label"]) for r in group_rows})
        if len(splits) > 1:
            cross_split_duplicate_groups += 1
        if len(labels) > 1:
            cross_label_duplicate_groups += 1
        duplicate_groups.append(
            {
                "sha256": sha256,
                "count": len(group_rows),
                "splits": splits,
                "labels": labels,
                "files": [str(r["relpath"]) for r in sorted(group_rows, key=lambda x: x["relpath"])],
            }
        )

    duplicate_groups.sort(key=lambda row: (-row["count"], row["sha256"]))
    total_images = len(records)
    decode_failed_rate = (len(decode_failed) / total_images) if total_images else 0.0

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "total_images": total_images,
        "decode_failed_count": len(decode_failed),
        "decode_failed_rate": decode_failed_rate,
        "decode_failed_examples": decode_failed[:25],
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "average_resolution": {
            "width": float(sum(widths) / len(widths)) if widths else 0.0,
            "height": float(sum(heights) / len(heights)) if heights else 0.0,
        },
        "mode_counts": dict(mode_counts),
        "ext_by_label": {label: dict(counter) for label, counter in ext_by_label.items()},
        "duplicate_summary": {
            "duplicate_groups": len(duplicate_groups),
            "duplicate_files_total": int(sum(group["count"] for group in duplicate_groups)),
            "cross_split_duplicate_groups": cross_split_duplicate_groups,
            "cross_label_duplicate_groups": cross_label_duplicate_groups,
        },
    }

    checks = [
        {
            "check": "decode_failed_rate_threshold",
            "ok": decode_failed_rate <= decode_failed_rate_threshold,
            "value": decode_failed_rate,
            "threshold": decode_failed_rate_threshold,
        },
        {
            "check": "cross_label_duplicate_groups",
            "ok": cross_label_duplicate_groups == 0,
            "value": cross_label_duplicate_groups,
            "expected": 0,
        },
    ]
    assertions = {
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }

    write_json(out_summary_path, summary)
    write_json(
        out_duplicates_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest_path),
            "groups": duplicate_groups,
        },
    )
    write_json(out_assertions_path, assertions)

    return {
        "summary_path": str(out_summary_path),
        "duplicates_path": str(out_duplicates_path),
        "assertions_path": str(out_assertions_path),
        "ok": assertions["ok"],
        "decode_failed_count": len(decode_failed),
        "duplicate_groups": len(duplicate_groups),
    }

