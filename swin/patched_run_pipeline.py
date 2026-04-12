from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run(name: str, cmd: List[str], cwd: Path) -> None:
    print(f"[patched] START {name}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)
    print(f"[patched] DONE  {name}", flush=True)


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
    print(f"[patched] dedup policy: {dedup.get('policy')}", flush=True)
    print(f"[patched] dedup stats_by_class keys: {sorted(stats.keys())}", flush=True)


def _assert_audit_cross_split_zero(audit_summary_path: Path, strict: bool) -> None:
    payload = _load_json(audit_summary_path)
    duplicate_summary = payload.get("duplicate_summary", {})
    cross_split = int(duplicate_summary.get("cross_split_duplicate_groups", -1))
    if strict and cross_split != 0:
        raise RuntimeError(
            "cross_split_duplicate_groups is not zero in audit summary. "
            f"value={cross_split}, path={audit_summary_path}"
        )
    mode = "STRICT" if strict else "WARN"
    print(f"[patched] audit cross_split_duplicate_groups={cross_split} ({mode})", flush=True)


def _assert_train_leakage_gate(preflight_path: Path) -> None:
    payload = _load_json(preflight_path)
    train_report = payload.get("train", {})
    checks = train_report.get("checks", [])
    gate = next((c for c in checks if c.get("check") == "pretrain_cross_split_leakage_gate"), None)
    if gate is None:
        raise RuntimeError(f"Leakage gate check not found in train preflight report: {preflight_path}")
    if not bool(gate.get("ok", False)):
        raise RuntimeError(f"Leakage gate check failed in preflight report: {gate}")

    first_counts = gate.get("first_scan_overlap_counts", {})
    final_counts = gate.get("final_scan_overlap_counts", {})
    if any(int(final_counts.get(key, 0)) != 0 for key in ["train_val", "train_test", "val_test"]):
        raise RuntimeError(f"Final overlap counts are non-zero after leakage gate: {final_counts}")

    print(f"[patched] leakage gate first overlaps={first_counts}", flush=True)
    print(f"[patched] leakage gate final overlaps={final_counts}", flush=True)


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
    print(f"[patched] label semantic check passed on {checked} rows", flush=True)


def _print_metric_summary(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics.json"
    classwise_path = run_dir / "classwise_metrics.json"
    calibration_path = run_dir / "calibration_report.json"

    metrics = _load_json(metrics_path)
    classwise = _load_json(classwise_path)
    calibration = _load_json(calibration_path)

    print("[patched] Final metrics summary", flush=True)
    print(
        (
            "  accuracy={accuracy:.6f} balanced_accuracy={balanced_accuracy:.6f} "
            "precision={precision:.6f} recall={recall:.6f} f1={f1:.6f} roc_auc={roc_auc:.6f} threshold={threshold:.4f}"
        ).format(**metrics),
        flush=True,
    )
    fake = classwise.get("fake", {})
    real = classwise.get("real", {})
    print(
        (
            "  fake: precision={:.6f} recall={:.6f} f1={:.6f} support={} | "
            "real: precision={:.6f} recall={:.6f} f1={:.6f} support={}"
        ).format(
            float(fake.get("precision", float("nan"))),
            float(fake.get("recall", float("nan"))),
            float(fake.get("f1", float("nan"))),
            int(fake.get("support", 0)),
            float(real.get("precision", float("nan"))),
            float(real.get("recall", float("nan"))),
            float(real.get("f1", float("nan"))),
            int(real.get("support", 0)),
        ),
        flush=True,
    )
    print(
        "  calibration: ece={:.6f} brier_score={:.6f}".format(
            float(calibration.get("ece", float("nan"))),
            float(calibration.get("brier_score", float("nan"))),
        ),
        flush=True,
    )


def _parse_seeds(raw: str | None) -> List[int]:
    if raw is None or not raw.strip():
        return []
    seeds: List[int] = []
    for token in raw.split(","):
        value = token.strip()
        if not value:
            continue
        seeds.append(int(value))
    # preserve order, remove duplicates
    deduped: List[int] = []
    seen = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        deduped.append(seed)
    return deduped


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


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: List[float], mean_value: float) -> float:
    if not values:
        return float("nan")
    return (sum((v - mean_value) ** 2 for v in values) / len(values)) ** 0.5


def _print_multi_seed_summary(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]
    print("[patched] Multi-seed aggregate summary", flush=True)
    for key in keys:
        values = [float(row["metrics"][key]) for row in results if key in row["metrics"]]
        if not values:
            continue
        mean_value = _mean(values)
        std_value = _std(values, mean_value)
        print(f"  {key}: mean={mean_value:.6f} std={std_value:.6f}", flush=True)
    print("[patched] Per-seed runs", flush=True)
    for row in results:
        print(
            f"  seed={row['seed']} run_dir={row['run_dir']} balanced_accuracy={float(row['metrics'].get('balanced_accuracy', float('nan'))):.6f}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run patched end-to-end pipeline in one command")
    parser.add_argument(
        "--config",
        default="configs/pipeline_full_swin_optimized.yaml",
        help="Pipeline config path",
    )
    parser.add_argument(
        "--model",
        choices=["swin", "efficientnet"],
        default="swin",
        help="Model to train/evaluate",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seeds for automatic multi-seed runs (e.g. 42,123,777)",
    )
    parser.add_argument(
        "--strict-audit-cross-split-zero",
        action="store_true",
        help="Fail if raw audit reports cross_split_duplicate_groups != 0",
    )
    parser.add_argument("--skip-audit", action="store_true", help="Skip raw audit rerun")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval stage")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = _repo_root()
    cfg_path = (root / args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    python_exe = sys.executable
    prepared_manifest = root / "data" / "prepared" / "deepdetect-2025" / "manifest.json"
    audit_summary = root / "data" / "processed" / "deepdetect_2025" / "audit_summary_v1.json"
    outputs_root = root / "outputs" / "runs"
    seeds = _parse_seeds(args.seeds)

    if seeds and args.skip_eval:
        raise RuntimeError("--seeds mode requires eval enabled (do not pass --skip-eval).")

    if seeds:
        python_exe = sys.executable
        multi_seed_results: List[Dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="patched_seed_cfgs_") as tmp_dir_raw:
            tmp_dir = Path(tmp_dir_raw)
            for seed in seeds:
                seeded_cfg = _write_seeded_config(cfg_path, seed=seed, out_dir=tmp_dir)
                print(f"[patched] ===== seed {seed} =====", flush=True)
                commands_seed: List[tuple[str, List[str]]] = [
                    (
                        f"prepare-force-seed-{seed}",
                        [python_exe, "-m", "pipeline.cli", "prepare", "--config", str(seeded_cfg), "--force"],
                    )
                ]

                if not args.skip_audit:
                    commands_seed.append(
                        (
                            f"audit-force-seed-{seed}",
                            [python_exe, "-m", "pipeline.cli", "audit", "--config", str(seeded_cfg), "--force"],
                        )
                    )

                commands_seed.append(
                    (
                        f"train-seed-{seed}",
                        [python_exe, "-m", "pipeline.cli", "train", "--config", str(seeded_cfg), "--model", args.model],
                    )
                )
                commands_seed.append(
                    (
                        f"eval-seed-{seed}",
                        [python_exe, "-m", "pipeline.cli", "eval", "--config", str(seeded_cfg), "--model", args.model],
                    )
                )

                if args.dry_run:
                    for name, cmd in commands_seed:
                        print(f"[patched][dry-run] {name}: {' '.join(cmd)}")
                    continue

                _run(*commands_seed[0], cwd=root)
                if not prepared_manifest.exists():
                    raise FileNotFoundError(f"Prepared manifest missing after prepare: {prepared_manifest}")
                _assert_manifest_dedup(prepared_manifest)

                command_index = 1
                if not args.skip_audit:
                    _run(*commands_seed[command_index], cwd=root)
                    command_index += 1
                    if not audit_summary.exists():
                        raise FileNotFoundError(f"Audit summary missing after audit: {audit_summary}")
                    _assert_audit_cross_split_zero(audit_summary, strict=args.strict_audit_cross_split_zero)

                _run(*commands_seed[command_index], cwd=root)
                command_index += 1

                latest_run = _latest_run_dir(outputs_root=outputs_root, model_name=args.model)
                preflight_path = latest_run / "preflight_report.json"
                if not preflight_path.exists():
                    raise FileNotFoundError(f"Missing preflight report for latest run: {preflight_path}")
                _assert_train_leakage_gate(preflight_path)

                _run(*commands_seed[command_index], cwd=root)
                latest_run = _latest_run_dir(outputs_root=outputs_root, model_name=args.model)
                eval_predictions = latest_run / "eval_predictions.csv"
                if not eval_predictions.exists():
                    raise FileNotFoundError(f"Missing eval predictions file: {eval_predictions}")
                _assert_label_semantics(eval_predictions)
                _print_metric_summary(latest_run)

                metrics_path = latest_run / "metrics.json"
                multi_seed_results.append(
                    {
                        "seed": seed,
                        "run_dir": str(latest_run),
                        "metrics": _load_json(metrics_path),
                    }
                )

        if args.dry_run:
            return 0

        _print_multi_seed_summary(multi_seed_results)
        print("[patched] Multi-seed pipeline completed successfully.", flush=True)
        return 0

    commands: List[tuple[str, List[str]]] = [
        (
            "prepare-force",
            [python_exe, "-m", "pipeline.cli", "prepare", "--config", str(cfg_path), "--force"],
        )
    ]

    if not args.skip_audit:
        commands.append(
            (
                "audit-force",
                [python_exe, "-m", "pipeline.cli", "audit", "--config", str(cfg_path), "--force"],
            )
        )

    commands.append(
        (
            "train",
            [python_exe, "-m", "pipeline.cli", "train", "--config", str(cfg_path), "--model", args.model],
        )
    )

    if not args.skip_eval:
        commands.append(
            (
                "eval",
                [python_exe, "-m", "pipeline.cli", "eval", "--config", str(cfg_path), "--model", args.model],
            )
        )

    if args.dry_run:
        for name, cmd in commands:
            print(f"[patched][dry-run] {name}: {' '.join(cmd)}")
        return 0

    _run(*commands[0], cwd=root)
    if not prepared_manifest.exists():
        raise FileNotFoundError(f"Prepared manifest missing after prepare: {prepared_manifest}")
    _assert_manifest_dedup(prepared_manifest)

    command_index = 1
    if not args.skip_audit:
        _run(*commands[command_index], cwd=root)
        command_index += 1
        if not audit_summary.exists():
            raise FileNotFoundError(f"Audit summary missing after audit: {audit_summary}")
        _assert_audit_cross_split_zero(audit_summary, strict=args.strict_audit_cross_split_zero)

    _run(*commands[command_index], cwd=root)
    command_index += 1

    latest_run = _latest_run_dir(outputs_root=outputs_root, model_name=args.model)
    preflight_path = latest_run / "preflight_report.json"
    if not preflight_path.exists():
        raise FileNotFoundError(f"Missing preflight report for latest run: {preflight_path}")
    _assert_train_leakage_gate(preflight_path)

    if not args.skip_eval:
        _run(*commands[command_index], cwd=root)
        latest_run = _latest_run_dir(outputs_root=outputs_root, model_name=args.model)
        eval_predictions = latest_run / "eval_predictions.csv"
        if not eval_predictions.exists():
            raise FileNotFoundError(f"Missing eval predictions file: {eval_predictions}")
        _assert_label_semantics(eval_predictions)
        _print_metric_summary(latest_run)

    print(f"[patched] Pipeline completed successfully. Latest run: {latest_run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
