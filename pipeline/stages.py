from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

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
from pipeline.metrics import compute_confusion, metrics_to_rows, save_confusion_matrix_png
from pipeline.models import evaluate_model, infer_model, train_model
from pipeline.preflight import (
    check_checkpoint_collision,
    check_class_folders,
    check_dependencies,
    check_non_empty_split,
    check_split_ratio,
    summarize_preflight,
)
from pipeline.video import download_and_extract


def _raw_dataset_dir(cfg: Dict[str, Any]) -> Path:
    return resolve_path(cfg, cfg["paths"]["raw_root"]) / cfg["data"]["dataset_version"]


def _prepared_dataset_dir(cfg: Dict[str, Any]) -> Path:
    return resolve_path(cfg, cfg["paths"]["prepared_root"]) / cfg["data"]["dataset_version"]


def _expected_raw_train_dir(cfg: Dict[str, Any]) -> Path:
    return _raw_dataset_dir(cfg) / cfg["data"]["raw_train_subdir"]


def _expected_raw_test_dir(cfg: Dict[str, Any]) -> Path:
    return _raw_dataset_dir(cfg) / cfg["data"]["raw_test_subdir"]


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


def _copy_images(paths: Iterable[Path], dst_dir: Path, prefix: str) -> List[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for idx, src in enumerate(paths):
        out_name = f"{prefix}_{idx:07d}{src.suffix.lower()}"
        dst = dst_dir / out_name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


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
    seed = int(cfg["project"]["seed"])

    checks = [
        check_split_ratio(val_ratio),
        check_class_folders(raw_train_dir, class_names),
        check_class_folders(raw_test_dir, class_names),
    ]
    if bool(cfg["prepare"]["augmentation"].get("enabled", False)):
        checks.append(check_dependencies("augmentation"))
    if with_video and video_urls:
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
    video_urls = video_urls or []
    if with_video and video_urls:
        video_raw_root = ensure_dir(resolve_path(cfg, cfg["paths"]["raw_root"]) / cfg["video"]["output_subdir"])
        download_dir = ensure_dir(video_raw_root / "downloads")
        extracted_dir = ensure_dir(video_raw_root / "extracted")
        for url in video_urls:
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
            video_stats.append(stats)

    split_sources: Dict[str, Dict[str, List[Path]]] = {"train": {}, "val": {}, "test": {}}
    source_hashes: Dict[str, str] = {}
    for class_name in class_names:
        raw_train_images = list_images(raw_train_dir / class_name)
        if with_video and class_name == "fake":
            video_images = list_images(resolve_path(cfg, cfg["paths"]["raw_root"]) / cfg["video"]["output_subdir"] / "extracted")
            raw_train_images.extend(video_images)
            raw_train_images = sorted(raw_train_images)

        train_paths, val_paths = _split_paths(raw_train_images, val_ratio=val_ratio, seed=seed)
        test_paths = list_images(raw_test_dir / class_name)
        split_sources["train"][class_name] = train_paths
        split_sources["val"][class_name] = val_paths
        split_sources["test"][class_name] = test_paths
        source_hashes[class_name] = metadata_hash(raw_train_images + test_paths, base_root=_raw_dataset_dir(cfg))

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
        "class_names": class_names,
        "source_hashes": source_hashes,
        "split_counts": split_counts,
        "video_enrichment": {
            "enabled": with_video,
            "urls": video_urls,
            "stats": video_stats,
        },
        "augmentation": {
            "enabled": bool(aug_cfg.get("enabled", False)),
            "target_class": aug_cfg.get("target_class"),
            "generated_count": aug_generated,
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
    outputs_root = ensure_dir(resolve_path(cfg, cfg["paths"]["outputs_root"]))
    run_dir = run_dir or create_run_dir(outputs_root, model_name=model_name, tag=cfg["artifacts"]["tag"])
    prepared_root = _prepared_dataset_dir(cfg)

    dump_yaml(run_dir / "config_resolved.yaml", cfg)
    copy_if_exists(prepared_root / "manifest.json", run_dir / "data_manifest_snapshot.json")

    preflight = _build_train_eval_preflight(cfg, model_name, run_dir)
    _update_preflight_report(run_dir, "train", preflight)
    _require_preflight_ok(preflight, "train")

    train_summary = train_model(model_name=model_name, cfg=cfg, prepared_root=prepared_root, run_dir=run_dir)
    write_json(run_dir / "train_summary.json", train_summary)
    return {"run_dir": str(run_dir), "train_summary": train_summary}


def run_eval(
    cfg: Dict[str, Any],
    model_name: str,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
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
    skip_prepare: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    skip_infer: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if not skip_setup:
        result["setup"] = run_setup(cfg=cfg, force=False)
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
