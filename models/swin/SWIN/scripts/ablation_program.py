from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: List[str], cwd: Path, name: str) -> None:
    print(f"[ablation] START {name}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)
    print(f"[ablation] DONE  {name}", flush=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _confusion(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, int]:
    pred = (probs >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    c = _confusion(y_true, probs, threshold)
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    fake_recall = tp / (tp + fn + 1e-12)
    real_recall = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    bal_acc = 0.5 * (fake_recall + real_recall)
    acc = (tp + tn) / max(1, len(y_true))
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision_fake": float(precision),
        "recall_fake": float(fake_recall),
        "recall_real": float(real_recall),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def _ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (n / total) * abs(conf - acc)
    return float(ece)


@dataclass
class SplitData:
    y_true: np.ndarray
    probs: np.ndarray


def _load_eval_predictions(pred_path: Path) -> SplitData:
    rows = _read_csv_rows(pred_path)
    y_true = np.array([int(r["true_label"]) for r in rows], dtype=int)
    probs = np.array([float(r["prob_fake"]) for r in rows], dtype=float)
    return SplitData(y_true=y_true, probs=probs)


def _load_val_predictions_from_run(run_dir: Path) -> Tuple[SplitData, str]:
    val_pred_path = run_dir / "val_predictions.csv"
    if val_pred_path.exists():
        rows = _read_csv_rows(val_pred_path)
        y_true = np.array([int(r["true_label"]) for r in rows], dtype=int)
        probs = np.array([float(r["prob_fake"]) for r in rows], dtype=float)
        return SplitData(y_true=y_true, probs=probs), "val_predictions"

    eval_pred_path = run_dir / "eval_predictions.csv"
    if eval_pred_path.exists():
        rows = _read_csv_rows(eval_pred_path)
        y_true = np.array([int(r["true_label"]) for r in rows], dtype=int)
        probs = np.array([float(r["prob_fake"]) for r in rows], dtype=float)
        return SplitData(y_true=y_true, probs=probs), "eval_predictions_fallback"

    raise RuntimeError(f"Missing prediction artifacts for A1/A2 in {run_dir}")


def _threshold_grid() -> np.ndarray:
    return np.linspace(0.05, 0.95, 19)


def _policy_threshold(policy: str, val: SplitData) -> float:
    y = val.y_true
    p = val.probs
    ts = _threshold_grid()

    if policy == "fixed_0.50":
        return 0.5
    if policy == "fixed_0.60":
        return 0.6

    best_t = 0.5
    best_s = -1.0

    if policy == "best_bal_acc":
        for t in ts:
            m = _metrics(y, p, float(t))
            s = m["balanced_accuracy"]
            if s > best_s:
                best_s = s
                best_t = float(t)
        return best_t

    if policy == "best_f1_fake":
        for t in ts:
            m = _metrics(y, p, float(t))
            rec = m["recall_fake"]
            prec = m["precision_fake"]
            f1 = (2.0 * rec * prec) / (rec + prec + 1e-12)
            if f1 > best_s:
                best_s = f1
                best_t = float(t)
        return best_t

    if policy.startswith("constrained_real_at_fake_"):
        floor = float(policy.split("_")[-1]) / 100.0
        best_s = -1.0
        for t in ts:
            m = _metrics(y, p, float(t))
            if m["recall_fake"] < floor:
                continue
            if m["recall_real"] > best_s:
                best_s = m["recall_real"]
                best_t = float(t)
        return best_t

    if policy == "youden_j":
        for t in ts:
            m = _metrics(y, p, float(t))
            j = m["recall_fake"] + m["recall_real"] - 1.0
            if j > best_s:
                best_s = j
                best_t = float(t)
        return best_t

    if policy.startswith("cost_fp"):
        # policy format cost_fp3_fn1 or cost_fp5_fn1
        fp_cost = float(policy.split("_")[1].replace("fp", ""))
        fn_cost = float(policy.split("_")[2].replace("fn", ""))
        best_cost = float("inf")
        for t in ts:
            c = _confusion(y, p, float(t))
            cost = fp_cost * c["fp"] + fn_cost * c["fn"]
            if cost < best_cost:
                best_cost = cost
                best_t = float(t)
        return best_t

    raise ValueError(f"Unknown threshold policy: {policy}")


def _fit_calibrator(name: str, val: SplitData):
    y = val.y_true
    logits = _logit(val.probs)

    if name == "none":
        return lambda z: z

    if name == "temperature":
        temp = 1.0
        best_loss = float("inf")
        for t in np.linspace(0.5, 5.0, 91):
            p = _sigmoid(logits / t)
            eps = 1e-12
            loss = -float(np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            if loss < best_loss:
                best_loss = loss
                temp = float(t)
        return lambda z: z / temp

    if name == "platt":
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(logits.reshape(-1, 1), y)
        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])
        return lambda z: a * z + b

    if name == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val.probs, y)
        return lambda z: _logit(np.clip(iso.predict(_sigmoid(z)), 1e-6, 1 - 1e-6))

    raise ValueError(f"Unknown calibrator: {name}")


def _run_a1_a2_for_run(run_dir: Path, out_dir: Path) -> None:
    test = _load_eval_predictions(run_dir / "eval_predictions.csv")
    val, val_source = _load_val_predictions_from_run(run_dir)

    policies = [
        "fixed_0.50",
        "fixed_0.60",
        "best_bal_acc",
        "best_f1_fake",
        "constrained_real_at_fake_95",
        "constrained_real_at_fake_98",
        "youden_j",
        "cost_fp3_fn1",
        "cost_fp5_fn1",
    ]
    calibrators = ["none", "temperature", "platt", "isotonic"]

    rows: List[Dict[str, Any]] = []
    for cal_name in calibrators:
        transform = _fit_calibrator(cal_name, val)
        val_logits = transform(_logit(val.probs))
        test_logits = transform(_logit(test.probs))
        val_probs = _sigmoid(val_logits)
        test_probs = _sigmoid(test_logits)

        for policy in policies:
            t = _policy_threshold(policy, SplitData(y_true=val.y_true, probs=val_probs))
            m = _metrics(test.y_true, test_probs, t)
            row = {
                "run_dir": str(run_dir),
                "calibrator": cal_name,
                "policy": policy,
                "val_source": val_source,
                "ece_test": _ece(test.y_true, test_probs),
            }
            row.update(m)
            rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "a1_a2_policy_calibration_comparison.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit(root: Path) -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), check=True, capture_output=True, text=True)
    return res.stdout.strip()


def _freeze_contract(root: Path, base_cfg: Path, seeds: List[int], out_dir: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "base_config": str(base_cfg.resolve()),
        "seed_list": seeds,
        "dataset_manifest": str((root / "data" / "prepared" / "deepdetect-2025" / "manifest.json").resolve()),
        "runner": "scripts/ablation_program.py",
    }
    _write_json(out_dir / "phase0_contract.json", payload)


def _write_registry_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    headers = [
        "run_id",
        "phase",
        "family",
        "factor_name",
        "factor_value",
        "seed",
        "val_protocol",
        "calibrator",
        "threshold_policy",
        "mean_primary_metric",
        "test_real_recall",
        "test_fake_recall",
        "test_fp",
        "test_tn",
        "test_auc",
        "test_pr_real",
        "test_ece",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)


def _append_registry_row(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "phase",
                "family",
                "factor_name",
                "factor_value",
                "seed",
                "val_protocol",
                "calibrator",
                "threshold_policy",
                "mean_primary_metric",
                "test_real_recall",
                "test_fake_recall",
                "test_fp",
                "test_tn",
                "test_auc",
                "test_pr_real",
                "test_ece",
                "notes",
            ],
        )
        writer.writerow(row)


def _split_csv_arg(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    seen: set[int] = set()
    for token in _split_csv_arg(raw):
        seed = int(token)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _gen_seed_cfg(base_cfg: Path, seed: int, val_protocol: str, out_path: Path) -> Path:
    payload = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
    payload.setdefault("project", {})["seed"] = int(seed)
    payload.setdefault("prepare", {})["validation_protocol"] = val_protocol
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_path


def _latest_run(outputs_root: Path, tag_contains: str) -> Path:
    cands = [p for p in outputs_root.iterdir() if p.is_dir() and tag_contains in p.name]
    if not cands:
        raise RuntimeError(f"No run found with tag {tag_contains}")
    return sorted(cands)[-1]


def run_a6(
    root: Path,
    base_cfg: Path,
    seeds: List[int],
    protocols: List[str],
    out_dir: Path,
    dry_run: bool,
    continue_on_error: bool,
) -> None:
    outputs_root = root / "outputs" / "runs"
    reg = out_dir / "ablation_registry.csv"
    for protocol in protocols:
        for seed in seeds:
            cfg_out = root / ".ablation_configs" / f"{out_dir.name}_a6_{protocol}_seed{seed}.yaml"
            _gen_seed_cfg(base_cfg, seed=seed, val_protocol=protocol, out_path=cfg_out)
            cmd = [
                sys.executable,
                "main.py",
                "--config",
                str(cfg_out),
                "--model",
                "swin",
                "--save-summary",
                str(out_dir / "summaries" / f"a6_{protocol}_seed{seed}.json"),
            ]
            if dry_run:
                print("[ablation][dry-run]", " ".join(cmd))
                continue
            try:
                _run(cmd, cwd=root, name=f"A6_{protocol}_seed{seed}")

                latest = _latest_run(outputs_root, "_swin_")
                rp = _read_json(latest / "run_report.json")
                row = {
                    "run_id": latest.name,
                    "phase": "A6",
                    "family": "validation_protocol",
                    "factor_name": "prepare.validation_protocol",
                    "factor_value": protocol,
                    "seed": seed,
                    "val_protocol": protocol,
                    "calibrator": str(rp.get("metrics", {}).get("temperature", "none")),
                    "threshold_policy": "from_eval_pipeline",
                    "mean_primary_metric": rp["classwise_metrics"]["real"]["recall"],
                    "test_real_recall": rp["classwise_metrics"]["real"]["recall"],
                    "test_fake_recall": rp["classwise_metrics"]["fake"]["recall"],
                    "test_fp": rp["confusion_counts"]["fp"],
                    "test_tn": rp["confusion_counts"]["tn"],
                    "test_auc": rp["metrics"]["roc_auc"],
                    "test_pr_real": rp["metrics"].get("pr_auc_real", float("nan")),
                    "test_ece": rp["calibration"]["ece"],
                    "notes": "A6 protocol comparison",
                }
                _append_registry_row(reg, row)
            except Exception as exc:
                fail_row = {
                    "run_id": "",
                    "phase": "A6",
                    "family": "validation_protocol",
                    "factor_name": "prepare.validation_protocol",
                    "factor_value": protocol,
                    "seed": seed,
                    "val_protocol": protocol,
                    "calibrator": "",
                    "threshold_policy": "",
                    "mean_primary_metric": "",
                    "test_real_recall": "",
                    "test_fake_recall": "",
                    "test_fp": "",
                    "test_tn": "",
                    "test_auc": "",
                    "test_pr_real": "",
                    "test_ece": "",
                    "notes": f"FAILED: {type(exc).__name__}: {exc}",
                }
                _append_registry_row(reg, fail_row)
                print(
                    f"[ablation] FAILED A6 protocol={protocol} seed={seed}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if not continue_on_error:
                    raise


def run_a1_a2(root: Path, target_run: Path, out_dir: Path) -> None:
    _run_a1_a2_for_run(target_run, out_dir / "a1_a2")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablation program executor (A6 prioritized + A1/A2).")
    p.add_argument("--base-config", default="configs/pipeline_full_swin_generalization_v2.yaml")
    p.add_argument("--target-run", default="outputs/runs/20260402_181141_swin_full_swin_gen_v2")
    p.add_argument("--out-dir", default="outputs/ablations")
    p.add_argument("--seeds", default="17,23,42", help="Comma-separated seed list for A6")
    p.add_argument(
        "--protocols",
        default="random,source_aware",
        help="Comma-separated prepare.validation_protocol values for A6",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining A6 combinations even after individual failures",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-a6", action="store_true")
    p.add_argument("--skip-a1a2", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = _repo_root()
    base_cfg = (root / args.base_config).resolve()
    target_run = (root / args.target_run).resolve()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (root / args.out_dir / f"batch_{batch_id}").resolve()

    seeds = _parse_seeds(args.seeds)
    protocols = _split_csv_arg(args.protocols)
    if not seeds:
        raise ValueError("--seeds produced an empty seed list")
    if not protocols:
        raise ValueError("--protocols produced an empty protocol list")

    _freeze_contract(root=root, base_cfg=base_cfg, seeds=seeds, out_dir=out_dir)
    _write_registry_header(out_dir / "ablation_registry.csv")

    if not args.skip_a6:
        run_a6(
            root=root,
            base_cfg=base_cfg,
            seeds=seeds,
            protocols=protocols,
            out_dir=out_dir,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )

    if not args.skip_a1a2 and not args.dry_run:
        run_a1_a2(root=root, target_run=target_run, out_dir=out_dir)

    print(f"[ablation] Batch completed: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
