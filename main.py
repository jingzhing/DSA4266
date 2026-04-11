from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _kaggle_token_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def _fmt_cmd(cmd: Iterable[str]) -> str:
    return " ".join(cmd)


def _run_step(name: str, cmd: List[str]) -> None:
    print(f"[main.py] START {name}: {_fmt_cmd(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[main.py] DONE  {name}", flush=True)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run_dir(outputs_root: Path, model_name: str) -> Path:
    candidates = [
        p for p in outputs_root.iterdir() if p.is_dir() and f"_{model_name}_" in p.name
    ]
    if not candidates:
        raise RuntimeError(f"No run directory found for model '{model_name}' in {outputs_root}")
    return sorted(candidates)[-1]


def _assert_manifest_dedup(manifest_path: Path) -> None:
    payload = _load_json(manifest_path)
    dedup = payload.get("deduplication")
    if not isinstance(dedup, dict):
        raise RuntimeError(f"Missing deduplication block in manifest: {manifest_path}")
    stats = dedup.get("stats_by_class")
    if not isinstance(stats, dict) or not stats:
        raise RuntimeError(f"Missing deduplication.stats_by_class in manifest: {manifest_path}")


def _assert_audit_cross_split_zero(audit_summary_path: Path, strict: bool) -> None:
    payload = _load_json(audit_summary_path)
    duplicate_summary = payload.get("duplicate_summary", {})
    cross_split = int(duplicate_summary.get("cross_split_duplicate_groups", -1))
    if strict and cross_split != 0:
        raise RuntimeError(
            "cross_split_duplicate_groups is not zero in audit summary. "
            f"value={cross_split}, path={audit_summary_path}"
        )


def _assert_train_leakage_gate(preflight_path: Path) -> None:
    payload = _load_json(preflight_path)
    train_report = payload.get("train", {})
    checks = train_report.get("checks", [])
    gate = next((c for c in checks if c.get("check") == "pretrain_cross_split_leakage_gate"), None)
    if gate is None:
        raise RuntimeError(f"Leakage gate check not found in train preflight report: {preflight_path}")
    if not bool(gate.get("ok", False)):
        raise RuntimeError(f"Leakage gate check failed in preflight report: {gate}")


def _assert_label_semantics(eval_predictions_path: Path) -> None:
    with eval_predictions_path.open("r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if not lines:
        raise RuntimeError(f"Empty predictions file: {eval_predictions_path}")

    header = lines[0].split(",")
    expected = ["path", "true_label", "prob_fake", "pred_label", "threshold"]
    if header != expected:
        raise RuntimeError(
            f"Unexpected predictions header in {eval_predictions_path}. got={header}, expected={expected}"
        )

    bad_fake = 0
    bad_real = 0
    checked = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 5:
            continue
        path, true_label = parts[0], parts[1]
        try:
            label_int = int(true_label)
        except ValueError:
            continue

        if "/test/fake/" in path:
            checked += 1
            if label_int != 1:
                bad_fake += 1
        elif "/test/real/" in path:
            checked += 1
            if label_int != 0:
                bad_real += 1

    if bad_fake or bad_real:
        raise RuntimeError(
            "Label semantic check failed in eval_predictions.csv: "
            f"bad_fake_labels={bad_fake}, bad_real_labels={bad_real}, checked={checked}"
        )


def _quality_gate(metrics_path: Path, classwise_path: Path, calibration_path: Path, min_real_recall: float, min_fake_recall: float, max_ece: float) -> Dict[str, Any]:
    metrics = _load_json(metrics_path)
    classwise = _load_json(classwise_path)
    calibration = _load_json(calibration_path)

    real_recall = float((classwise.get("real", {}) or {}).get("recall", float("nan")))
    fake_recall = float((classwise.get("fake", {}) or {}).get("recall", float("nan")))
    ece = float(calibration.get("ece", float("nan")))
    passed = real_recall >= min_real_recall and fake_recall >= min_fake_recall and ece <= max_ece

    return {
        "passed": bool(passed),
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "ece": ece,
        "balanced_accuracy": float(metrics.get("balanced_accuracy", float("nan"))),
        "roc_auc": float(metrics.get("roc_auc", float("nan"))),
    }


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_threshold_sweep(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "threshold": _safe_float(row.get("threshold")),
                "balanced_accuracy": _safe_float(row.get("balanced_accuracy")),
                "precision": _safe_float(row.get("precision")),
                "recall": _safe_float(row.get("recall")),
                "tn": _safe_float(row.get("tn"), 0.0),
                "fp": _safe_float(row.get("fp"), 0.0),
                "fn": _safe_float(row.get("fn"), 0.0),
                "tp": _safe_float(row.get("tp"), 0.0),
            })
    return rows


def _best_operating_point_for_constraint(sweep_rows: List[Dict[str, float]], min_fake_recall: float) -> Dict[str, Any]:
    best: Dict[str, Any] = {
        "found": False,
        "threshold": float("nan"),
        "real_recall": float("nan"),
        "fake_recall": float("nan"),
        "balanced_accuracy": float("nan"),
    }
    for row in sweep_rows:
        tp = row["tp"]
        fn = row["fn"]
        tn = row["tn"]
        fp = row["fp"]
        fake_recall = tp / (tp + fn + 1e-12)
        real_recall = tn / (tn + fp + 1e-12)
        if fake_recall < float(min_fake_recall):
            continue
        if (not best["found"]) or (real_recall > best["real_recall"]) or (
            real_recall == best["real_recall"] and row["balanced_accuracy"] > best["balanced_accuracy"]
        ):
            best = {
                "found": True,
                "threshold": float(row["threshold"]),
                "real_recall": float(real_recall),
                "fake_recall": float(fake_recall),
                "balanced_accuracy": float(row["balanced_accuracy"]),
            }
    return best


def _diagnose_quality_failure(
    run_dir: Path,
    quality: Dict[str, Any],
    min_real_recall: float,
    min_fake_recall: float,
    max_ece: float,
) -> Dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    classwise = _load_json(run_dir / "classwise_metrics.json") if (run_dir / "classwise_metrics.json").exists() else {}
    calibration = _load_json(run_dir / "calibration_report.json") if (run_dir / "calibration_report.json").exists() else {}
    confusion = _load_json(run_dir / "confusion_counts.json") if (run_dir / "confusion_counts.json").exists() else {}
    prob_summary = _load_json(run_dir / "probability_summary_by_class.json") if (run_dir / "probability_summary_by_class.json").exists() else {}
    sweep_rows = _read_threshold_sweep(run_dir / "threshold_sweep.csv")

    hypotheses: List[Dict[str, Any]] = []
    recommendations: List[str] = []

    real_recall = _safe_float((classwise.get("real", {}) or {}).get("recall"))
    fake_recall = _safe_float((classwise.get("fake", {}) or {}).get("recall"))
    ece = _safe_float(calibration.get("ece"))
    fp = int(confusion.get("fp", 0) or 0)
    fn = int(confusion.get("fn", 0) or 0)

    if real_recall < min_real_recall and fp > fn:
        hypotheses.append(
            {
                "id": "fp_dominant_boundary",
                "evidence": {
                    "real_recall": real_recall,
                    "fake_recall": fake_recall,
                    "fp": fp,
                    "fn": fn,
                },
                "summary": "Decision boundary is too permissive for fake predictions (FP-dominant failure).",
            }
        )
        recommendations.extend(
            [
                "Increase hard-negative mining intensity for real class (higher top_k/weight multiplier).",
                "Keep constrained thresholding with fake-recall floor and maximize real recall.",
            ]
        )

    if ece > max_ece:
        hypotheses.append(
            {
                "id": "miscalibration",
                "evidence": {
                    "ece": ece,
                    "max_ece": max_ece,
                },
                "summary": "Probabilities are poorly calibrated for operational decision-making.",
            }
        )
        recommendations.append("Increase calibration fitting budget or test isotonic regression fallback.")

    best_constrained = _best_operating_point_for_constraint(sweep_rows, min_fake_recall=min_fake_recall)
    if best_constrained.get("found", False):
        current_threshold = _safe_float(metrics.get("threshold"))
        suggested_threshold = _safe_float(best_constrained.get("threshold"))
        if abs(current_threshold - suggested_threshold) > 1e-9:
            hypotheses.append(
                {
                    "id": "suboptimal_operating_point",
                    "evidence": {
                        "current_threshold": current_threshold,
                        "suggested_threshold": suggested_threshold,
                        "suggested_real_recall": _safe_float(best_constrained.get("real_recall")),
                        "suggested_fake_recall": _safe_float(best_constrained.get("fake_recall")),
                    },
                    "summary": "Threshold in use may not be optimal under fake-recall constraint.",
                }
            )
            recommendations.append("Use constrained threshold selected from validation sweep artifacts.")

    true_real_mean = _safe_float((prob_summary.get("true_real", {}) or {}).get("mean"))
    true_fake_mean = _safe_float((prob_summary.get("true_fake", {}) or {}).get("mean"))
    if true_real_mean >= 0.5 and true_fake_mean >= 0.5:
        hypotheses.append(
            {
                "id": "poor_class_separation",
                "evidence": {
                    "mean_prob_true_real": true_real_mean,
                    "mean_prob_true_fake": true_fake_mean,
                },
                "summary": "Both classes are scored as fake-like, indicating representation/domain-shift issues.",
            }
        )
        recommendations.extend(
            [
                "Increase real-class diversity and apply symmetric robustness augmentations.",
                "Inspect top false positives and mine subtype clusters for targeted data enrichment.",
            ]
        )

    unique_recommendations = list(dict.fromkeys(recommendations))
    report = {
        "run_dir": str(run_dir),
        "quality_gate": quality,
        "constraints": {
            "min_real_recall": min_real_recall,
            "min_fake_recall": min_fake_recall,
            "max_ece": max_ece,
        },
        "hypotheses": hypotheses,
        "recommended_actions": unique_recommendations,
    }
    (run_dir / "failure_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_seeds(raw: str | None) -> List[int]:
    if raw is None or not raw.strip():
        return []
    seeds: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    out: List[int] = []
    seen = set()
    for s in seeds:
        if s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def _write_seeded_config(base_cfg_path: Path, seed: int, out_dir: Path) -> Path:
    payload = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
    project = payload.get("project")
    if not isinstance(project, dict):
        project = {}
        payload["project"] = project
    project["seed"] = int(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}.yaml"
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file robust runner for prepare->audit->train->eval with gates and optional multi-seed."
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline_full_swin_generalization_v2.yaml",
        help="Single config used for all stages.",
    )
    parser.add_argument(
        "--model",
        default="swin",
        choices=["swin", "efficientnet"],
        help="Model to train/evaluate.",
    )
    parser.add_argument("--seeds", default="", help="Comma-separated seeds for multi-seed execution.")
    parser.add_argument("--skip-audit", action="store_true", help="Skip raw audit rerun.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation stage.")
    parser.add_argument("--strict-audit-cross-split-zero", action="store_true", help="Fail if audit reports cross-split duplicates.")
    parser.add_argument("--min-real-recall", type=float, default=0.30, help="Quality gate: minimum real recall.")
    parser.add_argument("--min-fake-recall", type=float, default=0.95, help="Quality gate: minimum fake recall.")
    parser.add_argument("--max-ece", type=float, default=0.15, help="Quality gate: maximum expected calibration error.")
    parser.add_argument("--fail-on-quality-gate", action="store_true", help="Exit non-zero when quality gate fails.")
    parser.add_argument("--disable-auto-diagnosis", action="store_true", help="Disable automatic failure diagnosis artifact on quality-gate failure.")
    parser.add_argument(
        "--save-summary",
        default="outputs/logs/main_pipeline_summary.json",
        help="Path to save runner-level summary JSON.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print steps without executing.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = _repo_root()
    cfg_path = (root / args.config).resolve()
    _require_path(cfg_path, "Pipeline config")
    summary_path = (root / args.save_summary).resolve()
    outputs_root = (root / "outputs" / "runs").resolve()
    prepared_manifest = (root / "data" / "prepared" / "deepdetect-2025" / "manifest.json").resolve()
    audit_summary = (root / "data" / "processed" / "deepdetect_2025" / "audit_summary_v1.json").resolve()

    python_exe = sys.executable
    seeds = _parse_seeds(args.seeds)
    if seeds and args.skip_eval:
        raise RuntimeError("--seeds mode requires eval enabled.")

    run_summaries: List[Dict[str, Any]] = []

    def _run_one(active_cfg: Path, seed: int | None = None) -> Dict[str, Any]:
        steps: List[tuple[str, List[str]]] = [
            (
                "prepare",
                [python_exe, "-m", "pipeline.cli", "prepare", "--config", str(active_cfg), "--force"],
            )
        ]
        if not args.skip_audit:
            steps.append(
                (
                    "audit",
                    [python_exe, "-m", "pipeline.cli", "audit", "--config", str(active_cfg), "--force"],
                )
            )
        steps.append(
            (
                "train",
                [python_exe, "-u", "-m", "pipeline.cli", "train", "--config", str(active_cfg), "--model", args.model],
            )
        )
        if not args.skip_eval:
            steps.append(
                (
                    "eval",
                    [python_exe, "-m", "pipeline.cli", "eval", "--config", str(active_cfg), "--model", args.model],
                )
            )

        if args.dry_run:
            for name, cmd in steps:
                print(f"[main.py][dry-run] {name}: {_fmt_cmd(cmd)}")
            return {}

        _run_step(*steps[0])
        if not prepared_manifest.exists():
            raise FileNotFoundError(f"Prepared manifest missing after prepare: {prepared_manifest}")
        _assert_manifest_dedup(prepared_manifest)

        step_index = 1
        if not args.skip_audit:
            _run_step(*steps[step_index])
            step_index += 1
            if not audit_summary.exists():
                raise FileNotFoundError(f"Audit summary missing after audit: {audit_summary}")
            _assert_audit_cross_split_zero(audit_summary, strict=args.strict_audit_cross_split_zero)

        _run_step(*steps[step_index])
        step_index += 1

        latest_run = _latest_run_dir(outputs_root=outputs_root, model_name=args.model)
        preflight_path = latest_run / "preflight_report.json"
        if not preflight_path.exists():
            raise FileNotFoundError(f"Missing preflight report for latest run: {preflight_path}")
        _assert_train_leakage_gate(preflight_path)

        quality: Dict[str, Any] = {}
        diagnosis: Dict[str, Any] = {}
        if not args.skip_eval:
            _run_step(*steps[step_index])
            latest_run = _latest_run_dir(outputs_root=outputs_root, model_name=args.model)
            eval_predictions = latest_run / "eval_predictions.csv"
            metrics_path = latest_run / "metrics.json"
            classwise_path = latest_run / "classwise_metrics.json"
            calibration_path = latest_run / "calibration_report.json"

            if not eval_predictions.exists():
                raise FileNotFoundError(f"Missing eval predictions file: {eval_predictions}")
            _assert_label_semantics(eval_predictions)

            quality = _quality_gate(
                metrics_path=metrics_path,
                classwise_path=classwise_path,
                calibration_path=calibration_path,
                min_real_recall=float(args.min_real_recall),
                min_fake_recall=float(args.min_fake_recall),
                max_ece=float(args.max_ece),
            )
            print(
                (
                    f"[main.py] quality_gate passed={quality['passed']} "
                    f"real_recall={quality['real_recall']:.6f} "
                    f"fake_recall={quality['fake_recall']:.6f} "
                    f"ece={quality['ece']:.6f}"
                ),
                flush=True,
            )
            if not quality.get("passed", False):
                if not args.disable_auto_diagnosis:
                    diagnosis = _diagnose_quality_failure(
                        run_dir=latest_run,
                        quality=quality,
                        min_real_recall=float(args.min_real_recall),
                        min_fake_recall=float(args.min_fake_recall),
                        max_ece=float(args.max_ece),
                    )
                    print(
                        f"[main.py] failure diagnosis written: {latest_run / 'failure_diagnosis.json'}",
                        flush=True,
                    )
                if args.fail_on_quality_gate:
                    raise RuntimeError(
                        (
                            "Quality gate failed and --fail-on-quality-gate is enabled. "
                            f"real_recall={quality['real_recall']:.6f}, "
                            f"fake_recall={quality['fake_recall']:.6f}, ece={quality['ece']:.6f}"
                        )
                    )

        out = {
            "seed": seed,
            "config": str(active_cfg),
            "run_dir": str(latest_run),
            "quality_gate": quality,
            "failure_diagnosis": diagnosis,
        }
        return out

    if seeds:
        with tempfile.TemporaryDirectory(prefix="main_seed_cfgs_") as tmp_dir_raw:
            tmp_dir = Path(tmp_dir_raw)
            for seed in seeds:
                seeded_cfg = _write_seeded_config(cfg_path, seed=seed, out_dir=tmp_dir)
                print(f"[main.py] ===== seed {seed} =====", flush=True)
                run_summaries.append(_run_one(seeded_cfg, seed=seed))
    else:
        run_summaries.append(_run_one(cfg_path, seed=None))

    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "config": str(cfg_path),
            "model": args.model,
            "seeds": seeds,
            "quality_thresholds": {
                "min_real_recall": float(args.min_real_recall),
                "min_fake_recall": float(args.min_fake_recall),
                "max_ece": float(args.max_ece),
            },
            "runs": run_summaries,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"[main.py] Summary written: {summary_path}", flush=True)

    print("[main.py] Pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
