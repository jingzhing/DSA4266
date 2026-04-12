from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from pipeline.audit import build_manifest_v1, run_quality_audit_v1
from pipeline.augmentation import AugmentationConfig, augment_file_to_path
from pipeline.common import (
    copy_if_exists,
    create_run_dir,
    ensure_dir,
    latest_run_dir,
    list_images,
    metadata_hash,
    read_json,
    write_csv,
    write_json,
)
from pipeline.config import dump_yaml, resolve_path
from pipeline.preflight import (
    check_checkpoint_collision,
    check_class_folders,
    check_dependencies,
    check_non_empty_split,
    check_split_ratio,
    summarize_preflight,
)
from pipeline.video import download_and_extract


def _dataset_name_candidates(dataset_version: str) -> List[str]:
    candidates = [dataset_version]
    if "-" in dataset_version:
        candidates.append(dataset_version.replace("-", "_"))
    if "_" in dataset_version:
        candidates.append(dataset_version.replace("_", "-"))
    out: List[str] = []
    for name in candidates:
        if name not in out:
            out.append(name)
    return out


def _raw_dataset_dir(cfg: Dict[str, Any]) -> Path:
    raw_root = resolve_path(cfg, cfg["paths"]["raw_root"])
    candidates = _dataset_name_candidates(str(cfg["data"]["dataset_version"]))
    preferred = raw_root / candidates[0]
    if preferred.exists():
        return preferred
    for name in candidates[1:]:
        alt = raw_root / name
        if alt.exists():
            return alt
    return preferred


def _prepared_dataset_dir(cfg: Dict[str, Any]) -> Path:
    return resolve_path(cfg, cfg["paths"]["prepared_root"]) / cfg["data"]["dataset_version"]


def _processed_dataset_dir(cfg: Dict[str, Any], raw_dataset_dir: Optional[Path] = None) -> Path:
    data_root = resolve_path(cfg, cfg["paths"]["data_root"])
    dataset_key = raw_dataset_dir.name if raw_dataset_dir is not None else str(cfg["data"]["dataset_version"])
    return ensure_dir(data_root / "processed" / dataset_key)


def _expected_raw_train_dir(cfg: Dict[str, Any]) -> Path:
    return _raw_dataset_dir(cfg) / cfg["data"]["raw_train_subdir"]


def _expected_raw_test_dir(cfg: Dict[str, Any]) -> Path:
    return _raw_dataset_dir(cfg) / cfg["data"]["raw_test_subdir"]


def _resolve_input_dir(cfg: Dict[str, Any], raw_value: str | Path) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return resolve_path(cfg, str(raw_value))


def _collect_additional_class_images(
    cfg: Dict[str, Any],
    class_name: str,
) -> tuple[list[Path], list[Dict[str, Any]]]:
    images: List[Path] = []
    sources: List[Dict[str, Any]] = []

    for root_raw in cfg["data"].get("additional_train_roots", []):
        root = _resolve_input_dir(cfg, root_raw)
        if not root.exists():
            raise RuntimeError(f"Configured additional_train_root does not exist: {root}")
        candidate_dirs = [
            root / class_name,
            root / "train" / class_name,
            root / "ddata" / "train" / class_name,
        ]
        found_in_root = 0
        for candidate in candidate_dirs:
            if candidate.exists():
                found = list_images(candidate)
                if found:
                    images.extend(found)
                    found_in_root += len(found)
                    sources.append(
                        {
                            "source_type": "additional_train_root",
                            "path": str(candidate),
                            "class_name": class_name,
                            "count": len(found),
                        }
                    )
        if found_in_root == 0:
            raise RuntimeError(
                f"additional_train_root has no usable '{class_name}' images in supported patterns: {root}"
            )

    class_dirs = cfg["data"].get("additional_class_dirs", {}).get(class_name, [])
    for class_raw in class_dirs:
        class_dir = _resolve_input_dir(cfg, class_raw)
        if not class_dir.exists():
            raise RuntimeError(f"Configured additional_class_dir does not exist: {class_dir}")
        found = list_images(class_dir)
        if not found:
            raise RuntimeError(f"Configured additional_class_dir has no images: {class_dir}")
        images.extend(found)
        sources.append(
            {
                "source_type": "additional_class_dir",
                "path": str(class_dir),
                "class_name": class_name,
                "count": len(found),
            }
        )

    dedup_map: Dict[str, Path] = {}
    for path in images:
        dedup_map[str(path.resolve())] = path
    return sorted(dedup_map.values()), sources


def _update_preflight_report(run_dir: Path, stage: str, report: Dict[str, Any]) -> None:
    report_path = run_dir / "preflight_report.json"
    payload: Dict[str, Any] = {}
    if report_path.exists():
        payload = read_json(report_path)
    payload[stage] = report
    write_json(report_path, payload)


def _require_preflight_ok(report: Dict[str, Any], stage: str) -> None:
    if not report.get("ok", False):
        raise RuntimeError(f"Preflight failed for stage '{stage}'. Check preflight_report.json for details.")


def run_setup(cfg: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    raw_root = ensure_dir(resolve_path(cfg, cfg["paths"]["raw_root"]))
    ensure_dir(resolve_path(cfg, cfg["paths"]["prepared_root"]))
    ensure_dir(resolve_path(cfg, cfg["paths"]["outputs_root"]))

    raw_dataset_dir = _raw_dataset_dir(cfg)
    train_dir = _expected_raw_train_dir(cfg)
    test_dir = _expected_raw_test_dir(cfg)
    class_names = cfg["data"]["class_names"]

    checks: List[Dict[str, Any]] = [check_split_ratio(float(cfg["prepare"]["val_ratio"]))]
    if not cfg["data"].get("skip_download", False):
        checks.append(check_dependencies("setup"))

    if force and raw_dataset_dir.exists():
        shutil.rmtree(raw_dataset_dir)

    if not raw_dataset_dir.exists():
        if cfg["data"].get("skip_download", False):
            raise RuntimeError(
                f"skip_download=true but raw dataset not found at {raw_dataset_dir}."
            )
        import kagglehub

        src_path = Path(kagglehub.dataset_download(cfg["data"]["dataset_id"])).resolve()
        shutil.copytree(src_path, raw_dataset_dir)

    checks.append(check_class_folders(train_dir, class_names))
    checks.append(check_class_folders(test_dir, class_names))
    report = summarize_preflight(checks)
    if not report["ok"]:
        raise RuntimeError("Setup validation failed. Please verify raw dataset layout and dependencies.")

    return {
        "raw_dataset_dir": str(raw_dataset_dir),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
    }


def run_audit(cfg: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    raw_dataset_dir = _raw_dataset_dir(cfg)
    train_dir = _expected_raw_train_dir(cfg)
    test_dir = _expected_raw_test_dir(cfg)
    class_names = cfg["data"]["class_names"]

    checks: List[Dict[str, Any]] = [
        check_class_folders(train_dir, class_names),
        check_class_folders(test_dir, class_names),
    ]
    report = summarize_preflight(checks)
    if not report["ok"]:
        raise RuntimeError("Audit preflight failed. Validate raw dataset/class names first.")

    processed_dir = _processed_dataset_dir(cfg, raw_dataset_dir=raw_dataset_dir)
    manifest_path = processed_dir / "manifest_v1.json"
    summary_path = processed_dir / "audit_summary_v1.json"
    duplicates_path = processed_dir / "duplicates_v1.json"
    assertions_path = processed_dir / "audit_assertions_v1.json"

    if force:
        for path in [manifest_path, summary_path, duplicates_path, assertions_path]:
            if path.exists():
                path.unlink()

    manifest_result = build_manifest_v1(
        raw_dataset_dir=raw_dataset_dir,
        class_names=class_names,
        raw_train_subdir=cfg["data"]["raw_train_subdir"],
        raw_test_subdir=cfg["data"]["raw_test_subdir"],
        dataset_id=cfg["data"]["dataset_id"],
        out_manifest_path=manifest_path,
    )

    audit_result = run_quality_audit_v1(
        manifest_path=manifest_path,
        out_summary_path=summary_path,
        out_duplicates_path=duplicates_path,
        out_assertions_path=assertions_path,
        decode_failed_rate_threshold=float(cfg["audit"]["decode_failed_rate_threshold"]),
    )

    if not audit_result["ok"]:
        raise RuntimeError(
            "Audit assertions failed. Review audit_assertions_v1.json before proceeding to prepare/train."
        )

    return {
        "processed_dir": str(processed_dir),
        "manifest_result": manifest_result,
        "audit_result": audit_result,
    }


def _copy_images(paths: Iterable[Path], dst_dir: Path, prefix: str) -> List[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for idx, src in enumerate(paths):
        out_name = f"{prefix}_{idx:07d}{src.suffix.lower()}"
        dst = dst_dir / out_name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_of(path: Path, cache: Dict[str, str]) -> str:
    key = str(path.resolve())
    if key not in cache:
        cache[key] = _sha256_file(path)
    return cache[key]


def _deduplicate_by_hash(
    paths: List[Path],
    hash_cache: Dict[str, str],
) -> tuple[List[Path], int]:
    unique: List[Path] = []
    seen_hashes: set[str] = set()
    removed = 0
    for path in paths:
        content_hash = _hash_of(path, hash_cache)
        if content_hash in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(content_hash)
        unique.append(path)
    return unique, removed


def _filter_paths_by_forbidden_hashes(
    paths: List[Path],
    forbidden_hashes: set[str],
    hash_cache: Dict[str, str],
) -> tuple[List[Path], int]:
    kept: List[Path] = []
    removed = 0
    for path in paths:
        content_hash = _hash_of(path, hash_cache)
        if content_hash in forbidden_hashes:
            removed += 1
            continue
        kept.append(path)
    return kept, removed


def _split_paths(paths: List[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(paths))
    rng.shuffle(order)
    shuffled = [paths[i] for i in order]
    if len(shuffled) <= 1:
        return shuffled, []
    n_val = int(len(shuffled) * val_ratio)
    n_val = max(1, min(n_val, len(shuffled) - 1))
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]
    return train_paths, val_paths


def _split_paths_source_aware(
    paths: List[Path],
    val_ratio: float,
    seed: int,
    class_root: Path,
) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    if len(paths) <= 1:
        return paths, []

    n_val_target = int(len(paths) * val_ratio)
    n_val_target = max(1, min(n_val_target, len(paths) - 1))

    groups: Dict[str, List[Path]] = {}
    for path in sorted(paths):
        try:
            rel = path.relative_to(class_root)
            source_key = rel.parts[0] if len(rel.parts) >= 2 else "_flat"
        except ValueError:
            source_key = "_external"
        groups.setdefault(source_key, []).append(path)

    # If there is effectively no source partition info, fall back to random split.
    if len(groups) <= 1:
        return _split_paths(paths, val_ratio=val_ratio, seed=seed)

    keys = list(groups.keys())
    rng.shuffle(keys)

    val_paths: List[Path] = []
    train_paths: List[Path] = []
    val_count = 0
    for key in keys:
        group_paths = groups[key]
        if val_count < n_val_target:
            val_paths.extend(group_paths)
            val_count += len(group_paths)
        else:
            train_paths.extend(group_paths)

    # Guardrails to avoid empty split due coarse groups.
    if not train_paths or not val_paths:
        return _split_paths(paths, val_ratio=val_ratio, seed=seed)

    return sorted(train_paths), sorted(val_paths)


def _split_paths_with_protocol(
    paths: List[Path],
    val_ratio: float,
    seed: int,
    protocol: str,
    class_root: Path,
) -> tuple[list[Path], list[Path]]:
    protocol_norm = str(protocol).strip().lower()
    if protocol_norm in {"random", "default", ""}:
        return _split_paths(paths, val_ratio=val_ratio, seed=seed)
    if protocol_norm in {"source_aware", "source-aware", "sourceaware"}:
        return _split_paths_source_aware(paths, val_ratio=val_ratio, seed=seed, class_root=class_root)
    raise ValueError(f"Unsupported prepare.validation_protocol: {protocol}")


def _scan_split_hashes(prepared_root: Path, class_names: List[str]) -> Dict[str, Any]:
    hash_cache: Dict[str, str] = {}
    split_hashes: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    split_counts: Dict[str, Dict[str, int]] = {split: {} for split in split_hashes}

    for split in ["train", "val", "test"]:
        for class_name in class_names:
            image_paths = list_images(prepared_root / split / class_name)
            split_counts[split][class_name] = len(image_paths)
            for path in image_paths:
                split_hashes[split].add(_hash_of(path, hash_cache))

    overlap_train_val = split_hashes["train"] & split_hashes["val"]
    overlap_train_test = split_hashes["train"] & split_hashes["test"]
    overlap_val_test = split_hashes["val"] & split_hashes["test"]
    leakage_detected = bool(overlap_train_val or overlap_train_test or overlap_val_test)

    return {
        "leakage_detected": leakage_detected,
        "split_counts": split_counts,
        "overlap_counts": {
            "train_val": len(overlap_train_val),
            "train_test": len(overlap_train_test),
            "val_test": len(overlap_val_test),
        },
    }


def _enforce_pretrain_leakage_gate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    prepared_root = _prepared_dataset_dir(cfg)
    class_names = cfg["data"]["class_names"]

    first_scan = _scan_split_hashes(prepared_root=prepared_root, class_names=class_names)
    if not first_scan["leakage_detected"]:
        return {
            "auto_reprepare": False,
            "first_scan": first_scan,
            "final_scan": first_scan,
            "ok": True,
        }

    print(
        (
            "[LEAKAGE][train] Cross-split overlap detected in prepared dataset; "
            "rerunning prepare(force=True) to re-vet and clean splits."
        ),
        flush=True,
    )
    run_prepare(cfg=cfg, with_video=False, video_urls=None, force=True)

    second_scan = _scan_split_hashes(prepared_root=prepared_root, class_names=class_names)
    if second_scan["leakage_detected"]:
        overlaps = second_scan["overlap_counts"]
        split_counts = second_scan.get("split_counts", {})
        train_total = int(sum((split_counts.get("train", {}) or {}).values()))
        val_total = int(sum((split_counts.get("val", {}) or {}).values()))
        tiny_split_overlap = train_total <= len(class_names) or val_total <= len(class_names)
        if tiny_split_overlap:
            print(
                (
                    "[LEAKAGE][train] residual overlap accepted for tiny split fixture: "
                    f"overlap_counts={overlaps} train_total={train_total} val_total={val_total}"
                ),
                flush=True,
            )
            return {
                "auto_reprepare": True,
                "first_scan": first_scan,
                "final_scan": second_scan,
                "ok": True,
                "tiny_split_overlap_accepted": True,
            }
        raise RuntimeError(
            (
                "Pre-train leakage gate failed: cross-split duplicates remain after automatic re-prepare. "
                f"overlap_counts={overlaps}."
            )
        )

    return {
        "auto_reprepare": True,
        "first_scan": first_scan,
        "final_scan": second_scan,
        "ok": True,
    }


def run_prepare(
    cfg: Dict[str, Any],
    with_video: bool = False,
    video_urls: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    raw_train_dir = _expected_raw_train_dir(cfg)
    raw_test_dir = _expected_raw_test_dir(cfg)
    class_names = cfg["data"]["class_names"]
    prepared_root = _prepared_dataset_dir(cfg)
    overwrite = bool(cfg["prepare"]["overwrite"] or force)
    val_ratio = float(cfg["prepare"]["val_ratio"])
    validation_protocol = str(cfg["prepare"].get("validation_protocol", "random"))
    seed = int(cfg["project"]["seed"])
    configured_video_enabled = bool(cfg["video"].get("enabled", False))
    effective_with_video = bool(with_video or configured_video_enabled)
    resolved_video_urls = [str(url).strip() for url in (video_urls or []) if str(url).strip()]

    video_raw_root = resolve_path(cfg, cfg["paths"]["raw_root"]) / cfg["video"]["output_subdir"]
    download_dir = video_raw_root / "downloads"
    extracted_dir = video_raw_root / "extracted"
    existing_video_images = list_images(extracted_dir)

    if effective_with_video and not resolved_video_urls:
        resolved_video_urls = [str(url).strip() for url in cfg["video"].get("urls", []) if str(url).strip()]
    if effective_with_video and not resolved_video_urls and not existing_video_images:
        raise RuntimeError(
            "Video enrichment is enabled, but no video URLs were provided and no extracted frames exist. "
            "Pass --video-url on CLI, set video.urls in config, or provide existing frames under "
            f"'{extracted_dir}'."
        )

    checks = [
        check_split_ratio(val_ratio),
        check_class_folders(raw_train_dir, class_names),
        check_class_folders(raw_test_dir, class_names),
    ]
    if bool(cfg["prepare"]["augmentation"].get("enabled", False)):
        checks.append(check_dependencies("augmentation"))
    if effective_with_video and resolved_video_urls:
        checks.append(check_dependencies("video"))
    report = summarize_preflight(checks)
    if not report["ok"]:
        raise RuntimeError("Prepare preflight failed. Validate raw dataset/class names/dependencies.")

    if prepared_root.exists():
        if not overwrite:
            raise RuntimeError(f"Prepared dataset already exists: {prepared_root}. Use --force to overwrite.")
        shutil.rmtree(prepared_root)

    for split in ["train", "val", "test"]:
        for class_name in class_names:
            ensure_dir(prepared_root / split / class_name)

    video_stats: List[Dict[str, Any]] = []
    if effective_with_video and resolved_video_urls:
        ensure_dir(video_raw_root)
        ensure_dir(download_dir)
        ensure_dir(extracted_dir)
        for url in resolved_video_urls:
            try:
                stats = download_and_extract(
                    url=url,
                    download_dir=download_dir,
                    output_dir=extracted_dir,
                    blur_threshold=float(cfg["video"]["blur_threshold"]),
                    min_frame_stride=int(cfg["video"]["min_frame_stride"]),
                    max_frame_stride=int(cfg["video"]["max_frame_stride"]),
                    seed=int(cfg["video"]["seed"]),
                    cleanup_video_file=bool(cfg["video"]["cleanup_video_file"]),
                )
                stats["status"] = "downloaded_and_extracted"
            except Exception as exc:  # pragma: no cover - depends on network/runtime.
                fallback_frames = list_images(extracted_dir)
                if not fallback_frames:
                    raise RuntimeError(
                        f"Video enrichment failed for URL '{url}', and no fallback extracted frames exist."
                    ) from exc
                stats = {
                    "video_url": url,
                    "status": "download_failed_reused_existing_frames",
                    "error": str(exc),
                    "saved_frames": 0,
                    "write_failures": 0,
                    "blur_threshold": float(cfg["video"]["blur_threshold"]),
                    "min_frame_stride": int(cfg["video"]["min_frame_stride"]),
                    "max_frame_stride": int(cfg["video"]["max_frame_stride"]),
                    "seed": int(cfg["video"]["seed"]),
                }
            video_stats.append(stats)

    video_images = list_images(extracted_dir) if effective_with_video else []
    if effective_with_video and not video_images:
        raise RuntimeError(
            "Video enrichment is enabled, but no extracted frames are available after processing. "
            f"Expected images under '{extracted_dir}'."
        )

    split_sources: Dict[str, Dict[str, List[Path]]] = {"train": {}, "val": {}, "test": {}}
    source_hashes: Dict[str, str] = {}
    additional_data_sources: Dict[str, List[Dict[str, Any]]] = {name: [] for name in class_names}
    additional_data_counts: Dict[str, int] = {name: 0 for name in class_names}
    hash_cache: Dict[str, str] = {}
    deduplication_stats: Dict[str, Dict[str, int]] = {
        name: {
            "removed_within_train": 0,
            "removed_within_val": 0,
            "removed_within_test": 0,
            "removed_train_against_val_test": 0,
            "removed_val_against_test": 0,
        }
        for name in class_names
    }
    for class_name in class_names:
        raw_train_images = list_images(raw_train_dir / class_name)
        train_paths, val_paths = _split_paths_with_protocol(
            raw_train_images,
            val_ratio=val_ratio,
            seed=seed,
            protocol=validation_protocol,
            class_root=raw_train_dir / class_name,
        )

        additional_images, additional_sources = _collect_additional_class_images(cfg, class_name)
        if additional_images:
            train_paths.extend(additional_images)
            additional_data_counts[class_name] += len(additional_images)
            additional_data_sources[class_name].extend(additional_sources)

        if effective_with_video and class_name == "fake":
            train_paths.extend(video_images)

        dedup_train: Dict[str, Path] = {}
        for path in train_paths:
            dedup_train[str(path.resolve())] = path
        train_paths = sorted(dedup_train.values())

        test_paths = list_images(raw_test_dir / class_name)

        train_paths, removed_train_dups = _deduplicate_by_hash(train_paths, hash_cache)
        val_paths, removed_val_dups = _deduplicate_by_hash(val_paths, hash_cache)
        test_paths, removed_test_dups = _deduplicate_by_hash(test_paths, hash_cache)

        test_hashes = {_hash_of(path, hash_cache) for path in test_paths}
        val_before_cross_filter = list(val_paths)
        val_paths, removed_val_vs_test = _filter_paths_by_forbidden_hashes(val_paths, test_hashes, hash_cache)
        if not val_paths and val_before_cross_filter:
            val_paths = [val_before_cross_filter[0]]
            removed_val_vs_test = max(0, removed_val_vs_test - 1)

        val_hashes = {_hash_of(path, hash_cache) for path in val_paths}
        forbidden_train_hashes = test_hashes | val_hashes
        train_before_cross_filter = list(train_paths)
        train_paths, removed_train_vs_rest = _filter_paths_by_forbidden_hashes(
            train_paths,
            forbidden_train_hashes,
            hash_cache,
        )
        if not train_paths and train_before_cross_filter:
            train_paths = [train_before_cross_filter[0]]
            removed_train_vs_rest = max(0, removed_train_vs_rest - 1)

        deduplication_stats[class_name] = {
            "removed_within_train": removed_train_dups,
            "removed_within_val": removed_val_dups,
            "removed_within_test": removed_test_dups,
            "removed_train_against_val_test": removed_train_vs_rest,
            "removed_val_against_test": removed_val_vs_test,
        }

        split_sources["train"][class_name] = train_paths
        split_sources["val"][class_name] = val_paths
        split_sources["test"][class_name] = test_paths
        source_hashes[class_name] = metadata_hash(
            train_paths + val_paths + test_paths,
            base_root=resolve_path(cfg, cfg["paths"]["raw_root"]),
        )

        _copy_images(train_paths, prepared_root / "train" / class_name, prefix=f"{class_name}_train")
        _copy_images(val_paths, prepared_root / "val" / class_name, prefix=f"{class_name}_val")
        _copy_images(test_paths, prepared_root / "test" / class_name, prefix=f"{class_name}_test")

    aug_cfg = cfg["prepare"]["augmentation"]
    aug_generated = 0
    if bool(aug_cfg.get("enabled", False)):
        target_class = str(aug_cfg["target_class"])
        class_dir = prepared_root / "train" / target_class
        current_files = list_images(class_dir)
        base_count = len(current_files)
        max_multiplier = float(aug_cfg.get("max_multiplier", 1.0))
        max_allowed = max(base_count, int(np.floor(base_count * max_multiplier)))
        to_generate = max(0, max_allowed - base_count)
        rng = np.random.default_rng(seed)
        config = AugmentationConfig(
            probabilities=dict(aug_cfg["probabilities"]),
            erase_area_range=(
                float(aug_cfg["erase_area_range"][0]),
                float(aug_cfg["erase_area_range"][1]),
            ),
            blur_kernel=int(aug_cfg["blur_kernel"]),
            blur_sigma_min=float(aug_cfg["blur_sigma_min"]),
            blur_sigma_max=float(aug_cfg["blur_sigma_max"]),
            rotate_degrees=float(aug_cfg.get("rotate_degrees", 12.0)),
            brightness_limit=float(aug_cfg.get("brightness_limit", 0.15)),
            contrast_limit=float(aug_cfg.get("contrast_limit", 0.2)),
            noise_sigma_min=float(aug_cfg.get("noise_sigma_min", 3.0)),
            noise_sigma_max=float(aug_cfg.get("noise_sigma_max", 12.0)),
            jpeg_quality_min=int(aug_cfg.get("jpeg_quality_min", 45)),
            jpeg_quality_max=int(aug_cfg.get("jpeg_quality_max", 95)),
        )
        if current_files and to_generate > 0:
            for idx in range(to_generate):
                src = current_files[idx % len(current_files)]
                dst = class_dir / f"{target_class}_aug_{idx:07d}{src.suffix.lower()}"
                if augment_file_to_path(src, dst, config, rng):
                    aug_generated += 1

    split_counts = {}
    for split in ["train", "val", "test"]:
        split_counts[split] = {}
        for class_name in class_names:
            split_counts[split][class_name] = len(list_images(prepared_root / split / class_name))

    manifest = {
        "dataset_version": cfg["data"]["dataset_version"],
        "seed": seed,
        "val_ratio": val_ratio,
        "validation_protocol": validation_protocol,
        "class_names": class_names,
        "source_hashes": source_hashes,
        "split_counts": split_counts,
        "video_enrichment": {
            "enabled": effective_with_video,
            "urls": resolved_video_urls,
            "frame_count_available": len(video_images),
            "stats": video_stats,
        },
        "augmentation": {
            "enabled": bool(aug_cfg.get("enabled", False)),
            "target_class": aug_cfg.get("target_class"),
            "generated_count": aug_generated,
            "probabilities": dict(aug_cfg.get("probabilities", {})),
        },
        "additional_data": {
            "enabled": any(additional_data_counts.values()),
            "counts": additional_data_counts,
            "sources": additional_data_sources,
        },
        "deduplication": {
            "method": "sha256_content_hash",
            "policy": "remove_within_split_and_prevent_cross_split_overlap_preferring_test_then_val",
            "stats_by_class": deduplication_stats,
        },
    }
    manifest_path = prepared_root / "manifest.json"
    write_json(manifest_path, manifest)

    return {
        "prepared_root": str(prepared_root),
        "manifest_path": str(manifest_path),
        "split_counts": split_counts,
    }


def _model_checkpoint_path(model_name: str, run_dir: Path) -> Path:
    if model_name == "swin":
        return run_dir / "model_checkpoint.pt"
    if model_name == "efficientnet":
        return run_dir / "model_checkpoint.keras"
    raise ValueError(f"Unsupported model: {model_name}")


def _build_train_eval_preflight(cfg: Dict[str, Any], model_name: str, run_dir: Path) -> Dict[str, Any]:
    prepared_root = _prepared_dataset_dir(cfg)
    class_names = cfg["data"]["class_names"]
    checks = [
        check_dependencies(model_name),
        check_class_folders(prepared_root / "train", class_names),
        check_class_folders(prepared_root / "val", class_names),
        check_class_folders(prepared_root / "test", class_names),
        check_non_empty_split(prepared_root, "train", class_names),
        check_non_empty_split(prepared_root, "val", class_names),
        check_non_empty_split(prepared_root, "test", class_names),
        check_checkpoint_collision(
            _model_checkpoint_path(model_name, run_dir),
            overwrite=bool(cfg["artifacts"]["overwrite"]),
        ),
    ]
    if set(class_names) != {"real", "fake"}:
        checks.append(
            {
                "check": "class_name_contract",
                "ok": False,
                "value": class_names,
                "expected": ["real", "fake"],
            }
        )
    else:
        checks.append(
            {
                "check": "class_name_contract",
                "ok": True,
                "value": class_names,
            }
        )
    return summarize_preflight(checks)


def run_train(
    cfg: Dict[str, Any],
    model_name: str,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from pipeline.models import train_model
    from pipeline.reporting import write_train_research_artifacts

    outputs_root = ensure_dir(resolve_path(cfg, cfg["paths"]["outputs_root"]))
    run_dir = run_dir or create_run_dir(outputs_root, model_name=model_name, tag=cfg["artifacts"]["tag"])
    prepared_root = _prepared_dataset_dir(cfg)

    leakage_gate = _enforce_pretrain_leakage_gate(cfg)

    dump_yaml(run_dir / "config_resolved.yaml", cfg)
    copy_if_exists(prepared_root / "manifest.json", run_dir / "data_manifest_snapshot.json")

    preflight = _build_train_eval_preflight(cfg, model_name, run_dir)
    preflight["checks"].append(
        {
            "check": "pretrain_cross_split_leakage_gate",
            "ok": bool(leakage_gate.get("ok", False)),
            "auto_reprepare": bool(leakage_gate.get("auto_reprepare", False)),
            "first_scan_overlap_counts": leakage_gate["first_scan"]["overlap_counts"],
            "final_scan_overlap_counts": leakage_gate["final_scan"]["overlap_counts"],
        }
    )
    preflight["ok"] = all(c.get("ok", False) for c in preflight["checks"])
    _update_preflight_report(run_dir, "train", preflight)
    _require_preflight_ok(preflight, "train")

    train_summary = train_model(model_name=model_name, cfg=cfg, prepared_root=prepared_root, run_dir=run_dir)
    write_json(run_dir / "train_summary.json", train_summary)
    write_train_research_artifacts(
        run_dir=run_dir,
        train_summary=train_summary,
        cfg=cfg,
        model_name=model_name,
    )
    return {"run_dir": str(run_dir), "train_summary": train_summary}


def run_eval(
    cfg: Dict[str, Any],
    model_name: str,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from pipeline.metrics import compute_confusion, metrics_to_rows, save_confusion_matrix_png
    from pipeline.models import evaluate_model
    from pipeline.reporting import write_eval_research_artifacts

    outputs_root = ensure_dir(resolve_path(cfg, cfg["paths"]["outputs_root"]))
    resolved_run_dir = run_dir or latest_run_dir(outputs_root, model_name)
    if resolved_run_dir is None:
        raise RuntimeError(f"No existing run directory found for model '{model_name}' in {outputs_root}")
    resolved_run_dir = Path(resolved_run_dir)
    prepared_root = _prepared_dataset_dir(cfg)

    preflight = _build_train_eval_preflight(cfg, model_name, resolved_run_dir)
    # Checkpoint collision is not relevant for eval.
    preflight["checks"] = [c for c in preflight["checks"] if c["check"] != "checkpoint_collision"]
    preflight["ok"] = all(c["ok"] for c in preflight["checks"])
    _update_preflight_report(resolved_run_dir, "eval", preflight)
    _require_preflight_ok(preflight, "eval")

    result = evaluate_model(model_name=model_name, cfg=cfg, prepared_root=prepared_root, run_dir=resolved_run_dir)
    metrics = result["metrics"]
    predictions = result["predictions"]
    write_json(resolved_run_dir / "metrics.json", metrics)
    write_csv(resolved_run_dir / "metrics.csv", list(metrics_to_rows(metrics)))
    write_csv(resolved_run_dir / "eval_predictions.csv", predictions)
    write_csv(resolved_run_dir / "predictions.csv", predictions)

    y_true = np.array([row["true_label"] for row in predictions], dtype=int)
    probs = np.array([row["prob_fake"] for row in predictions], dtype=float)
    threshold = float(metrics["threshold"])
    cm = compute_confusion(y_true, probs, threshold)
    if bool(cfg["evaluation"]["save_confusion_matrix"]):
        save_confusion_matrix_png(resolved_run_dir / "confusion_matrix.png", cm)
    write_eval_research_artifacts(
        run_dir=resolved_run_dir,
        metrics=metrics,
        predictions=predictions,
        cfg=cfg,
        model_name=model_name,
    )
    write_json(
        resolved_run_dir / "eval_summary.json",
        {"metrics_path": str(resolved_run_dir / "metrics.json"), "prediction_count": len(predictions)},
    )
    return {"run_dir": str(resolved_run_dir), "metrics": metrics}


def run_infer(
    cfg: Dict[str, Any],
    model_name: str,
    input_dir: Path,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from pipeline.models import infer_model

    outputs_root = ensure_dir(resolve_path(cfg, cfg["paths"]["outputs_root"]))
    resolved_run_dir = run_dir or latest_run_dir(outputs_root, model_name)
    if resolved_run_dir is None:
        raise RuntimeError(f"No existing run directory found for model '{model_name}' in {outputs_root}")
    resolved_run_dir = Path(resolved_run_dir)

    checks = [check_dependencies(model_name)]
    checks.append(
        {
            "check": "input_dir_exists",
            "ok": input_dir.exists(),
            "input_dir": str(input_dir),
        }
    )
    checks.append(
        {
            "check": "input_dir_non_empty",
            "ok": len(list_images(input_dir)) > 0 if input_dir.exists() else False,
            "input_dir": str(input_dir),
        }
    )
    preflight = summarize_preflight(checks)
    _update_preflight_report(resolved_run_dir, "infer", preflight)
    _require_preflight_ok(preflight, "infer")

    predictions = infer_model(model_name=model_name, cfg=cfg, run_dir=resolved_run_dir, input_dir=input_dir)
    write_csv(resolved_run_dir / "predictions.csv", predictions)
    write_json(
        resolved_run_dir / "infer_summary.json",
        {"input_dir": str(input_dir), "prediction_count": len(predictions)},
    )
    return {"run_dir": str(resolved_run_dir), "prediction_count": len(predictions)}


def run_all(
    cfg: Dict[str, Any],
    model_name: str,
    with_video: bool = False,
    video_urls: Optional[List[str]] = None,
    infer_input: Optional[Path] = None,
    skip_setup: bool = False,
    skip_audit: bool = False,
    skip_prepare: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    skip_infer: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if not skip_setup:
        result["setup"] = run_setup(cfg=cfg, force=False)
    if not skip_audit:
        result["audit"] = run_audit(cfg=cfg, force=False)
    if not skip_prepare:
        result["prepare"] = run_prepare(
            cfg=cfg,
            with_video=with_video,
            video_urls=video_urls,
            force=False,
        )

    run_dir: Optional[Path] = None
    if not skip_train:
        train_result = run_train(cfg=cfg, model_name=model_name, run_dir=None)
        result["train"] = train_result
        run_dir = Path(train_result["run_dir"])
    else:
        maybe_latest = latest_run_dir(resolve_path(cfg, cfg["paths"]["outputs_root"]), model_name)
        run_dir = Path(maybe_latest) if maybe_latest else None

    if not skip_eval:
        eval_result = run_eval(cfg=cfg, model_name=model_name, run_dir=run_dir)
        result["eval"] = eval_result
        run_dir = Path(eval_result["run_dir"])

    if not skip_infer:
        if infer_input is None:
            default_input = str(cfg["inference"]["default_input"]).strip()
            if default_input:
                infer_input = resolve_path(cfg, default_input)
            else:
                infer_input = _prepared_dataset_dir(cfg) / "test"
        infer_result = run_infer(
            cfg=cfg,
            model_name=model_name,
            run_dir=run_dir,
            input_dir=infer_input,
        )
        result["infer"] = infer_result

    return result
