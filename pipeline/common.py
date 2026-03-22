from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_images(root: str | Path) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(
        p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def metadata_hash(paths: Iterable[Path], base_root: Path | None = None) -> str:
    sha = hashlib.sha256()
    for path in sorted(paths):
        if base_root:
            try:
                rel = str(path.relative_to(base_root))
            except ValueError:
                rel = str(path.resolve())
        else:
            rel = str(path)
        stat = path.stat()
        sha.update(rel.encode("utf-8"))
        sha.update(str(stat.st_size).encode("utf-8"))
        sha.update(str(int(stat.st_mtime)).encode("utf-8"))
    return sha.hexdigest()


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latest_run_dir(outputs_root: Path, model_name: str) -> Path | None:
    if not outputs_root.exists():
        return None
    candidates = [
        p
        for p in outputs_root.iterdir()
        if p.is_dir() and f"_{model_name}_" in p.name
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def create_run_dir(outputs_root: Path, model_name: str, tag: str) -> Path:
    run_name = f"{timestamp_id()}_{model_name}_{tag}"
    return ensure_dir(outputs_root / run_name)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())
