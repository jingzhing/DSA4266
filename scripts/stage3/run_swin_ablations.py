from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.config import load_config
from pipeline.stages import run_eval, run_prepare, run_train


def _validate_gpu_runtime(require_cuda: bool) -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        if require_cuda:
            raise RuntimeError("PyTorch is required for CUDA validation but is not importable.") from exc
        info = {
            "torch_importable": False,
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "nvidia_smi": "torch import failed",
        }
        print(json.dumps({"gpu_validation": info}, indent=2), flush=True)
        return info

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count() if cuda_available else 0)
    device_name = str(torch.cuda.get_device_name(0)) if cuda_available and device_count > 0 else None

    nvidia_smi_out = "nvidia-smi not found"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            nvidia_smi_out = result.stdout.strip()
        elif result.stderr.strip():
            nvidia_smi_out = result.stderr.strip()
    except Exception:
        pass

    info = {
        "torch_importable": True,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "nvidia_smi": nvidia_smi_out,
    }
    print(json.dumps({"gpu_validation": info}, indent=2), flush=True)

    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is not available. In Colab, set Runtime > Change runtime type > Hardware accelerator = GPU, then rerun."
        )
    return info


def _deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_default_experiments(base_dataset_version: str) -> List[Dict[str, Any]]:
    return [
        {
            "name": "baseline_full_finetune",
            "description": "Patched mapping retrain baseline.",
            "overrides": {
                "training": {
                    "optimization": {
                        "swin": {
                            "train_mode": "full_finetune",
                        }
                    }
                }
            },
        },
        {
            "name": "linear_probe",
            "description": "Head-only fine-tuning for shift robustness baseline.",
            "overrides": {
                "training": {
                    "optimization": {
                        "swin": {
                            "train_mode": "linear_probe",
                        }
                    }
                }
            },
        },
        {
            "name": "staged_unfreeze_h1",
            "description": "Head-only for 1 epoch, then full finetune.",
            "overrides": {
                "training": {
                    "optimization": {
                        "swin": {
                            "train_mode": "staged_unfreeze",
                            "staged_unfreeze_head_epochs": 1,
                        }
                    }
                }
            },
        },
        {
            "name": "staged_unfreeze_h2",
            "description": "Head-only for 2 epochs, then full finetune.",
            "overrides": {
                "training": {
                    "optimization": {
                        "swin": {
                            "train_mode": "staged_unfreeze",
                            "staged_unfreeze_head_epochs": 2,
                        }
                    }
                }
            },
        },
        {
            "name": "lr_5e5",
            "description": "Lower LR to improve threshold transfer stability.",
            "overrides": {
                "models": {"swin": {"lr": 5e-5}},
                "training": {
                    "optimization": {
                        "swin": {
                            "train_mode": "full_finetune",
                        }
                    }
                },
            },
        },
        {
            "name": "wd_003",
            "description": "Higher weight decay for regularization under distribution shift.",
            "overrides": {
                "training": {
                    "optimization": {
                        "swin": {
                            "train_mode": "full_finetune",
                            "weight_decay": 0.03,
                        }
                    }
                }
            },
        },
        {
            "name": "warmup0",
            "description": "No warmup to test optimizer sensitivity.",
            "overrides": {
                "training": {
                    "optimization": {
                        "swin": {
                            "train_mode": "full_finetune",
                            "warmup_epochs": 0,
                        }
                    }
                }
            },
        },
        {
            "name": "aug_mult_110_fake_only",
            "description": "Data-policy ablation: fake-only augmentation with lower multiplier.",
            "prepare_overrides": {
                "data": {"dataset_version": f"{base_dataset_version}-abl-aug110"},
                "prepare": {
                    "overwrite": True,
                    "augmentation": {
                        "enabled": True,
                        "target_class": "fake",
                        "max_multiplier": 1.10,
                    },
                },
            },
            "overrides": {
                "training": {
                    "optimization": {
                        "swin": {
                            "train_mode": "full_finetune",
                        }
                    }
                }
            },
        },
    ]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_run(run_dir: Path) -> Dict[str, Any]:
    train_summary = _read_json(run_dir / "train_summary.json")
    metrics = _read_json(run_dir / "metrics.json")

    threshold_sweep_path = run_dir / "threshold_sweep.json"
    threshold_transfer_gap = None
    bal_acc_gap = None
    best_test_threshold = None
    best_test_bal_acc = None
    if threshold_sweep_path.exists():
        rows = _read_json(threshold_sweep_path)
        if rows:
            best = max(rows, key=lambda row: float(row.get("balanced_accuracy", float("-inf"))))
            best_test_threshold = float(best.get("threshold"))
            best_test_bal_acc = float(best.get("balanced_accuracy"))
            threshold_transfer_gap = abs(float(train_summary["best_threshold"]) - best_test_threshold)
            bal_acc_gap = abs(float(train_summary["best_val_balanced_accuracy"]) - best_test_bal_acc)

    run_report_path = run_dir / "run_report.json"
    ece = None
    brier_score = None
    if run_report_path.exists():
        run_report = _read_json(run_report_path)
        calibration = run_report.get("calibration", {})
        ece = calibration.get("ece")
        brier_score = calibration.get("brier_score")

    return {
        "run_dir": str(run_dir.resolve()),
        "train": {
            "best_threshold": train_summary.get("best_threshold"),
            "best_val_balanced_accuracy": train_summary.get("best_val_balanced_accuracy"),
            "best_val_auc": train_summary.get("best_val_auc"),
            "best_epoch": train_summary.get("best_epoch"),
            "optimizer": train_summary.get("optimizer", {}),
        },
        "test": {
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "threshold": metrics.get("threshold"),
            "best_test_threshold": best_test_threshold,
            "best_test_balanced_accuracy": best_test_bal_acc,
        },
        "shift": {
            "threshold_transfer_gap": threshold_transfer_gap,
            "val_test_bal_acc_gap": bal_acc_gap,
            "ece": ece,
            "brier_score": brier_score,
        },
    }


def _markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# SWIN Ablation Summary",
        "",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        f"- Base config: `{payload['base_config']}`",
        f"- Experiment count: `{len(payload['experiments'])}`",
        "",
        "| Experiment | Train Mode | Val BalAcc | Test BalAcc | Test ROC-AUC | Fake Recall | Thr Transfer Gap | BalAcc Gap | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in payload["experiments"]:
        summary = exp.get("summary", {})
        train = summary.get("train", {})
        test = summary.get("test", {})
        shift = summary.get("shift", {})
        opt = train.get("optimizer", {})
        train_mode = opt.get("train_mode", "-")

        def _f(value: Any, digits: int = 4) -> str:
            if value is None:
                return "-"
            try:
                return f"{float(value):.{digits}f}"
            except Exception:
                return str(value)

        lines.append(
            "| "
            + f"{exp['name']} | {train_mode} | "
            + f"{_f(train.get('best_val_balanced_accuracy'))} | "
            + f"{_f(test.get('balanced_accuracy'))} | "
            + f"{_f(test.get('roc_auc'))} | "
            + f"{_f(test.get('recall'))} | "
            + f"{_f(shift.get('threshold_transfer_gap'))} | "
            + f"{_f(shift.get('val_test_bal_acc_gap'))} | "
            + f"{_f(shift.get('ece'))} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Selection",
            "",
            f"- fake_recall_floor: `{payload['selection']['rule']['fake_recall_floor']}`",
            f"- max_ece: `{payload['selection']['rule']['max_ece']}`",
            f"- max_val_test_bal_acc_gap: `{payload['selection']['rule']['max_val_test_bal_acc_gap']}`",
            f"- eligible_count: `{payload['selection']['eligible_count']}`",
            f"- selected_experiment: `{payload['selection']['selected_experiment'] or '-'} `",
            "",
            "## Notes",
            "",
            "- `threshold_transfer_gap` is |best_val_threshold - best_test_threshold_from_sweep|.",
            "- Lower `val_test_bal_acc_gap` and lower ECE indicate better validation-test transfer.",
            "- Use this table to choose a deployment candidate after confirming stable fake recall and acceptable false positives.",
            "",
        ]
    )
    return "\n".join(lines)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _enforce_fake_only_augmentation(cfg: Dict[str, Any]) -> None:
    prepare = cfg.get("prepare", {})
    aug = prepare.get("augmentation", {})
    if not bool(aug.get("enabled", False)):
        raise RuntimeError(
            "Ablation policy requires augmentation enabled for train/fake. "
            "Got prepare.augmentation.enabled=False."
        )
    target = str(aug.get("target_class", "")).strip().lower()
    if target != "fake":
        raise RuntimeError(
            "Ablation policy requires fake-only augmentation. "
            f"Got prepare.augmentation.target_class={aug.get('target_class')}"
        )


def _select_candidate(
    experiments: List[Dict[str, Any]],
    fake_recall_floor: float,
    max_ece: float,
    max_val_test_bal_acc_gap: float,
) -> Dict[str, Any]:
    eligible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for experiment in experiments:
        summary = experiment.get("summary", {})
        test = summary.get("test", {})
        shift = summary.get("shift", {})
        recall = _as_float(test.get("recall"))
        bal_acc = _as_float(test.get("balanced_accuracy"))
        roc_auc = _as_float(test.get("roc_auc"))
        ece = _as_float(shift.get("ece"))
        gap = _as_float(shift.get("val_test_bal_acc_gap"))

        reasons: List[str] = []
        if recall is None or recall < fake_recall_floor:
            reasons.append("fake_recall_floor")
        if ece is None or ece > max_ece:
            reasons.append("max_ece")
        if gap is None or gap > max_val_test_bal_acc_gap:
            reasons.append("max_val_test_bal_acc_gap")
        if bal_acc is None:
            reasons.append("test_balanced_accuracy_missing")
        if roc_auc is None:
            reasons.append("test_roc_auc_missing")

        record = {
            "name": experiment.get("name"),
            "test_balanced_accuracy": bal_acc,
            "test_roc_auc": roc_auc,
            "fake_recall": recall,
            "ece": ece,
            "val_test_bal_acc_gap": gap,
        }
        if reasons:
            record["reasons"] = reasons
            rejected.append(record)
        else:
            eligible.append(record)

    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (
                float(row["test_balanced_accuracy"]),
                float(row["test_roc_auc"]),
                float(row["fake_recall"]),
                -float(row["ece"]),
                -float(row["val_test_bal_acc_gap"]),
            ),
            reverse=True,
        )[0]

    return {
        "rule": {
            "fake_recall_floor": fake_recall_floor,
            "max_ece": max_ece,
            "max_val_test_bal_acc_gap": max_val_test_bal_acc_gap,
        },
        "eligible_count": len(eligible),
        "selected_experiment": selected["name"] if selected else None,
        "selected_summary": selected,
        "eligible": eligible,
        "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWIN retrain + ablations and summarize val-test shift diagnostics.")
    parser.add_argument("--config", default="configs/pipeline_full_swin_optimized.yaml", help="Base pipeline config")
    parser.add_argument("--out-json", default="docs/SWIN_ABLATION_SUMMARY.json", help="Output JSON summary")
    parser.add_argument("--out-md", default="docs/SWIN_ABLATION_SUMMARY.md", help="Output markdown summary")
    parser.add_argument(
        "--experiments-json",
        default="",
        help="Optional custom experiment list JSON path. If omitted, built-in defaults are used.",
    )
    parser.add_argument(
        "--skip-prepare-for-data-policy",
        action="store_true",
        help="Skip prepare reruns for experiments containing prepare_overrides.",
    )
    parser.add_argument(
        "--fake-recall-floor",
        type=float,
        default=0.85,
        help="Minimum fake recall required for deployment-candidate eligibility.",
    )
    parser.add_argument(
        "--max-ece",
        type=float,
        default=0.20,
        help="Maximum calibration error (ECE) allowed for eligibility.",
    )
    parser.add_argument(
        "--max-val-test-balacc-gap",
        type=float,
        default=0.15,
        help="Maximum allowed absolute gap between best val bal-acc and best test bal-acc-from-sweep.",
    )
    parser.add_argument(
        "--enforce-fake-only-augmentation",
        action="store_true",
        help="Fail fast if augmentation policy is not enabled and targeted to fake class.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast when CUDA is unavailable (recommended for Colab GPU runs).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print experiment plan only")
    args = parser.parse_args()

    gpu_info = _validate_gpu_runtime(require_cuda=bool(args.require_cuda))

    base_cfg = load_config(args.config)
    base_dataset_version = str(base_cfg["data"]["dataset_version"])

    if args.experiments_json:
        experiments = _read_json(Path(args.experiments_json))
        if not isinstance(experiments, list):
            raise ValueError("Custom experiments JSON must be a list of objects.")
    else:
        experiments = build_default_experiments(base_dataset_version=base_dataset_version)

    if args.dry_run:
        print(json.dumps({"config": args.config, "experiments": experiments}, indent=2))
        return

    results: List[Dict[str, Any]] = []
    for idx, experiment in enumerate(experiments, start=1):
        exp_name = str(experiment["name"])
        print(f"[ablation] ({idx}/{len(experiments)}) starting: {exp_name}", flush=True)

        cfg = copy.deepcopy(base_cfg)
        cfg["artifacts"] = dict(cfg["artifacts"])
        cfg["artifacts"]["tag"] = f"swin_ablation_{exp_name}"
        cfg["artifacts"]["overwrite"] = False

        prepare_overrides = experiment.get("prepare_overrides")
        if prepare_overrides and not args.skip_prepare_for_data_policy:
            prepare_cfg = copy.deepcopy(cfg)
            _deep_update(prepare_cfg, prepare_overrides)
            run_prepare(prepare_cfg, with_video=False, video_urls=None, force=True)
            cfg = prepare_cfg

        _deep_update(cfg, experiment.get("overrides", {}))

        if args.enforce_fake_only_augmentation:
            _enforce_fake_only_augmentation(cfg)

        train_result = run_train(cfg, model_name="swin")
        run_dir = Path(train_result["run_dir"])
        eval_result = run_eval(cfg, model_name="swin", run_dir=run_dir)
        summary = _summarize_run(run_dir)
        summary["eval_metrics"] = eval_result.get("metrics", {})

        results.append(
            {
                "name": exp_name,
                "description": experiment.get("description", ""),
                "run_dir": str(run_dir.resolve()),
                "config_overrides": experiment.get("overrides", {}),
                "prepare_overrides": prepare_overrides or {},
                "summary": summary,
            }
        )
        print(f"[ablation] completed: {exp_name} -> {run_dir}", flush=True)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": str(Path(args.config).resolve()),
        "gpu_validation": gpu_info,
        "experiments": results,
    }
    payload["selection"] = _select_candidate(
        experiments=results,
        fake_recall_floor=float(args.fake_recall_floor),
        max_ece=float(args.max_ece),
        max_val_test_bal_acc_gap=float(args.max_val_test_balacc_gap),
    )

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md = Path(args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown(payload), encoding="utf-8")

    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
