from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from pipeline.metrics import (
    classwise_recalls_from_counts,
    compute_binary_metrics,
    confusion_counts,
    find_best_threshold,
    find_threshold_max_real_recall,
    pr_auc_fake_and_real,
    sigmoid,
)


def _binary_label_from_class_idx(class_idx: np.ndarray, fake_idx: int) -> np.ndarray:
    return (class_idx.astype(int) == int(fake_idx)).astype(int)


def _resolve_fake_idx_from_class_to_idx(class_to_idx: Dict[str, int], context: str) -> int:
    if "fake" not in class_to_idx:
        raise RuntimeError(f"Missing 'fake' class in {context} class_to_idx: {class_to_idx}")
    return int(class_to_idx["fake"])


def _load_image_paths_for_inference(input_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise RuntimeError(f"No images found in input directory: {input_dir}")
    return image_paths


def _cfg_get(cfg: Dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _resolve_flag(value: Any, auto_value: bool) -> bool:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return auto_value
    return bool(value)


def _make_grad_scaler(torch_module: Any, enabled: bool):
    amp_module = getattr(torch_module, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler("cuda", enabled=enabled)
    return torch_module.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(torch_module: Any, enabled: bool):
    amp_module = getattr(torch_module, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        return amp_module.autocast(device_type="cuda", enabled=enabled)
    return torch_module.cuda.amp.autocast(enabled=enabled)


def _robust_train_transform(transforms_module: Any, img_size: int):
    return transforms_module.Compose(
        [
            transforms_module.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms_module.RandomHorizontalFlip(p=0.5),
            transforms_module.RandomApply([transforms_module.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            transforms_module.RandomApply([transforms_module.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.35),
            transforms_module.RandomApply([transforms_module.RandomAdjustSharpness(sharpness_factor=1.8)], p=0.2),
            transforms_module.ToTensor(),
            transforms_module.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def _build_weighted_sampler(torch_module: Any, sample_weights: np.ndarray):
    weights = torch_module.as_tensor(sample_weights, dtype=torch_module.double)
    return torch_module.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _fit_temperature_scaler(
    logits: np.ndarray,
    labels: np.ndarray,
    max_steps: int = 200,
    lr: float = 0.05,
) -> float:
    import torch

    if logits.size == 0:
        return 1.0

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_temp = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.Adam([log_temp], lr=float(lr))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(max(1, int(max_steps))):
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        scaled_logits = logits_t / temperature
        loss = loss_fn(scaled_logits, labels_t)
        loss.backward()
        optimizer.step()

    temperature = float(torch.exp(log_temp.detach()).cpu().item())
    return float(np.clip(temperature, 1e-3, 100.0))


def train_model(model_name: str, cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    if model_name == "swin":
        return _train_swin(cfg, prepared_root, run_dir)
    if model_name == "efficientnet":
        return _train_efficientnet(cfg, prepared_root, run_dir)
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model(model_name: str, cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    if model_name == "swin":
        return _evaluate_swin(cfg, prepared_root, run_dir)
    if model_name == "efficientnet":
        return _evaluate_efficientnet(cfg, prepared_root, run_dir)
    raise ValueError(f"Unsupported model: {model_name}")


def infer_model(
    model_name: str,
    cfg: Dict[str, Any],
    run_dir: Path,
    input_dir: Path,
) -> List[Dict[str, Any]]:
    if model_name == "swin":
        return _infer_swin(cfg, run_dir, input_dir)
    if model_name == "efficientnet":
        return _infer_efficientnet(cfg, run_dir, input_dir)
    raise ValueError(f"Unsupported model: {model_name}")


def _train_swin(cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    import timm

    model_cfg = cfg["models"]["swin"]
    seed = int(cfg["project"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dir = prepared_root / "train"
    val_dir = prepared_root / "val"
    swin_opt = _cfg_get(cfg, ["training", "optimization", "swin"], {})
    dataloader_opt = _cfg_get(cfg, ["training", "optimization", "dataloader"], {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_threads = int(swin_opt.get("num_threads", 0))
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    advanced_cfg = _cfg_get(cfg, ["training", "advanced"], {})
    robust_aug_cfg = _cfg_get(cfg, ["training", "advanced", "robust_augmentation"], {})
    use_robust_aug = bool(robust_aug_cfg.get("enabled", False))

    if use_robust_aug:
        train_tf = _robust_train_transform(transforms, int(model_cfg["img_size"]))
    elif bool(cfg.get("training", {}).get("runtime_augmentation", False)):
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(model_cfg["img_size"], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize((model_cfg["img_size"], model_cfg["img_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((model_cfg["img_size"], model_cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    train_eval_ds = ImageFolder(str(train_dir), transform=eval_tf)
    val_ds = ImageFolder(str(val_dir), transform=eval_tf)
    class_names = list(train_ds.classes)
    fake_idx = _resolve_fake_idx_from_class_to_idx(train_ds.class_to_idx, context="train")
    val_fake_idx = _resolve_fake_idx_from_class_to_idx(val_ds.class_to_idx, context="val")
    if fake_idx != val_fake_idx:
        raise RuntimeError(
            (
                "Inconsistent fake class index between train and val splits: "
                f"train={fake_idx}, val={val_fake_idx}. "
                f"train_class_to_idx={train_ds.class_to_idx}, val_class_to_idx={val_ds.class_to_idx}"
            )
        )

    num_workers = int(model_cfg["num_workers"])
    pin_memory = _resolve_flag(dataloader_opt.get("pin_memory", "auto"), auto_value=(device == "cuda"))
    persistent_workers = _resolve_flag(
        dataloader_opt.get("persistent_workers", "auto"),
        auto_value=(num_workers > 0),
    )
    prefetch_factor = max(2, int(dataloader_opt.get("prefetch_factor", 2)))
    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    hard_neg_cfg = _cfg_get(cfg, ["training", "advanced", "hard_negative_mining"], {})
    hard_neg_enabled = bool(hard_neg_cfg.get("enabled", False))
    hard_neg_top_k = max(1, int(hard_neg_cfg.get("top_k", 2048)))
    hard_neg_weight_multiplier = float(hard_neg_cfg.get("real_weight_multiplier", 3.0))
    hard_neg_min_epoch = max(1, int(hard_neg_cfg.get("min_epoch_to_start", 2)))
    hard_neg_update_every = max(1, int(hard_neg_cfg.get("update_every_epochs", 1)))
    sample_weights = np.ones(len(train_ds), dtype=np.float64)

    def _make_train_loader() -> Any:
        if hard_neg_enabled:
            sampler = _build_weighted_sampler(torch, sample_weights)
            return DataLoader(
                train_ds,
                batch_size=model_cfg["batch_size"],
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **loader_kwargs,
            )
        return DataLoader(
            train_ds,
            batch_size=model_cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **loader_kwargs,
        )

    train_loader = _make_train_loader()
    val_loader = DataLoader(
        val_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )

    model = timm.create_model(model_cfg["model_name"], pretrained=True, num_classes=1).to(device)
    base_lr = float(model_cfg["lr"])
    weight_decay = float(swin_opt.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    use_amp = bool(swin_opt.get("amp", False)) and device == "cuda"
    scaler = _make_grad_scaler(torch, enabled=use_amp)
    gradient_accumulation_steps = max(1, int(swin_opt.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(swin_opt.get("max_grad_norm", 0.0))
    warmup_epochs = max(0, int(swin_opt.get("warmup_epochs", 0)))
    min_lr = float(swin_opt.get("min_lr", 1e-6))
    scheduler_name = str(swin_opt.get("scheduler", "none")).strip().lower()
    scheduler = None
    total_epochs = int(model_cfg["epochs"])
    if scheduler_name == "cosine":
        t_max = max(1, total_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    elif scheduler_name not in {"none", "off", ""}:
        raise ValueError(f"Unsupported SWIN scheduler: {scheduler_name}")

    early_stopping_patience = max(0, int(swin_opt.get("early_stopping_patience", 0)))
    early_stopping_min_delta = float(swin_opt.get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    batch_log_cfg = _cfg_get(cfg, ["training", "debug", "batch_logging"], {})
    batch_log_enabled = bool(batch_log_cfg.get("enabled", False))
    batch_log_every_n_steps = max(1, int(batch_log_cfg.get("every_n_steps", 1)))
    batch_log_interval_seconds = max(0.0, float(batch_log_cfg.get("interval_seconds", 1.0)))

    constrained_cfg = _cfg_get(cfg, ["training", "advanced", "selection"], {})
    constrained_enabled = bool(constrained_cfg.get("enabled", False))
    min_fake_recall = float(constrained_cfg.get("min_fake_recall", 0.95))

    best_val_score = -1.0
    best_threshold = float(cfg["training"]["default_threshold"])
    best_auc = float("nan")
    best_epoch = -1
    best_real_recall = -1.0
    best_fake_recall = -1.0
    best_feasible = False
    checkpoint_path = run_dir / "model_checkpoint.pt"
    history: List[Dict[str, Any]] = []
    print(
        (
            f"[SWIN][train] run_dir={run_dir} device={device} "
            f"train_samples={len(train_ds)} val_samples={len(val_ds)} "
            f"batch_size={model_cfg['batch_size']} steps_per_epoch={len(train_loader)} "
            f"num_workers={num_workers} num_threads={num_threads} "
            f"scheduler={scheduler_name} warmup_epochs={warmup_epochs} "
            f"grad_accum={gradient_accumulation_steps} max_grad_norm={max_grad_norm}"
        ),
        flush=True,
    )

    for epoch in range(total_epochs):
        epoch_start = time.perf_counter()
        last_batch_log_time = epoch_start - batch_log_interval_seconds
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = base_lr * float(epoch + 1) / float(warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr

        model.train()
        losses: List[float] = []
        running_loss = 0.0
        optimizer_steps = 0
        epoch_total_steps = len(train_loader)
        epoch_samples = len(train_ds)
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y_bin = (y.numpy() == fake_idx).astype(np.float32)
            y_bin_t = torch.from_numpy(y_bin).to(device, non_blocking=True)

            with _autocast_context(torch, enabled=use_amp):
                logits = model(x).squeeze(1)
                raw_loss = loss_fn(logits, y_bin_t)
                loss = raw_loss / gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (step_idx % gradient_accumulation_steps == 0) or (step_idx == len(train_loader))
            if should_step:
                if max_grad_norm > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            losses.append(float(raw_loss.item()))
            running_loss += float(raw_loss.item())

            if batch_log_enabled:
                now = time.perf_counter()
                time_gate_open = (now - last_batch_log_time) >= batch_log_interval_seconds
                step_gate_open = step_idx % batch_log_every_n_steps == 0
                is_boundary_step = step_idx == 1 or step_idx == epoch_total_steps
                if is_boundary_step or (time_gate_open and step_gate_open):
                    elapsed_sec = max(0.0, now - epoch_start)
                    sec_per_step = elapsed_sec / step_idx if step_idx > 0 else float("nan")
                    steps_remaining = max(0, epoch_total_steps - step_idx)
                    eta_sec = sec_per_step * steps_remaining if step_idx > 0 else float("nan")
                    avg_loss = running_loss / step_idx if step_idx > 0 else float("nan")
                    samples_done = min(epoch_samples, step_idx * int(model_cfg["batch_size"]))
                    print(
                        (
                            f"[SWIN][batch] epoch={epoch + 1}/{total_epochs} "
                            f"step={step_idx}/{epoch_total_steps} "
                            f"samples={samples_done}/{epoch_samples} "
                            f"loss={float(raw_loss.item()):.6f} "
                            f"avg_loss={avg_loss:.6f} "
                            f"lr={float(optimizer.param_groups[0]['lr']):.8f} "
                            f"opt_steps={optimizer_steps} "
                            f"elapsed_sec={elapsed_sec:.2f} "
                            f"eta_sec={eta_sec:.2f}"
                        ),
                        flush=True,
                    )
                    last_batch_log_time = now

        val_logits: List[np.ndarray] = []
        val_y_raw: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                logits = model(x).squeeze(1).detach().cpu().numpy()
                val_logits.append(logits)
                val_y_raw.append(y.numpy())

        y_raw = np.concatenate(val_y_raw).astype(int)
        y_bin = _binary_label_from_class_idx(y_raw, fake_idx=fake_idx)
        probs = sigmoid(np.concatenate(val_logits))
        if constrained_enabled:
            threshold, constrained_payload = find_threshold_max_real_recall(
                y_true=y_bin,
                probs_fake=probs,
                min_fake_recall=min_fake_recall,
            )
            score = float(constrained_payload["balanced_accuracy"])
            feasible = bool(constrained_payload["feasible"] >= 0.5)
            current_real_recall = float(constrained_payload["real_recall"])
            current_fake_recall = float(constrained_payload["fake_recall"])
        else:
            threshold, score = find_best_threshold(y_bin, probs)
            counts_tmp = confusion_counts(y_bin, probs, threshold)
            recalls_tmp = classwise_recalls_from_counts(counts_tmp)
            feasible = True
            current_real_recall = float(recalls_tmp["real_recall"])
            current_fake_recall = float(recalls_tmp["fake_recall"])

        metrics = compute_binary_metrics(y_bin, probs, threshold)
        pr_metrics = pr_auc_fake_and_real(y_bin, probs)
        train_loss_mean = float(np.mean(losses) if losses else float("nan"))
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss_mean,
                "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "val_roc_auc": float(metrics["roc_auc"]),
                "val_pr_auc_fake": float(pr_metrics["pr_auc_fake"]),
                "val_pr_auc_real": float(pr_metrics["pr_auc_real"]),
                "val_real_recall": current_real_recall,
                "val_fake_recall": current_fake_recall,
                "constraint_feasible": bool(feasible),
                "threshold": float(threshold),
                "lr": current_lr,
            }
        )
        epoch_time_sec = float(time.perf_counter() - epoch_start)

        improved = False
        if constrained_enabled:
            improved = (
                (feasible and not best_feasible)
                or (
                    feasible == best_feasible
                    and current_real_recall > (best_real_recall + early_stopping_min_delta)
                )
                or (
                    feasible == best_feasible
                    and abs(current_real_recall - best_real_recall) <= early_stopping_min_delta
                    and score > (best_val_score + early_stopping_min_delta)
                )
            )
        else:
            improved = score > (best_val_score + early_stopping_min_delta)

        if improved:
            best_val_score = score
            best_threshold = threshold
            best_auc = metrics["roc_auc"]
            best_epoch = epoch + 1
            best_real_recall = current_real_recall
            best_fake_recall = current_fake_recall
            best_feasible = feasible
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg,
                    "class_names": class_names,
                    "fake_idx": fake_idx,
                    "best_threshold": best_threshold,
                    "best_val_balanced_accuracy": best_val_score,
                    "best_val_auc": best_auc,
                    "selection": {
                        "constrained_enabled": constrained_enabled,
                        "min_fake_recall": min_fake_recall,
                        "best_real_recall": best_real_recall,
                        "best_fake_recall": best_fake_recall,
                        "best_feasible": best_feasible,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None and epoch + 1 > warmup_epochs:
            scheduler.step()

        print(
            (
                f"[SWIN][epoch {epoch + 1}/{total_epochs}] "
                f"loss={train_loss_mean:.6f} "
                f"val_bal_acc={float(metrics['balanced_accuracy']):.6f} "
                f"val_auc={float(metrics['roc_auc']):.6f} "
                f"val_real_recall={current_real_recall:.6f} "
                f"val_fake_recall={current_fake_recall:.6f} "
                f"threshold={float(threshold):.4f} "
                f"lr={current_lr:.8f} "
                f"improved={improved} "
                f"time_sec={epoch_time_sec:.2f}"
            ),
            flush=True,
        )

        if (
            hard_neg_enabled
            and (epoch + 1) >= hard_neg_min_epoch
            and ((epoch + 1 - hard_neg_min_epoch) % hard_neg_update_every == 0)
        ):
            train_logits: List[np.ndarray] = []
            train_y_raw: List[np.ndarray] = []
            model.eval()
            with torch.no_grad():
                for x_eval, y_eval in train_eval_loader:
                    x_eval = x_eval.to(device, non_blocking=True)
                    logits_eval = model(x_eval).squeeze(1).detach().cpu().numpy()
                    train_logits.append(logits_eval)
                    train_y_raw.append(y_eval.numpy())

            train_probs = sigmoid(np.concatenate(train_logits))
            train_raw_idx = np.concatenate(train_y_raw).astype(int)
            is_real = (train_raw_idx != fake_idx)
            real_indices = np.where(is_real)[0]
            if real_indices.size > 0:
                real_probs = train_probs[real_indices]
                n_pick = min(int(hard_neg_top_k), int(real_indices.size))
                hardest_pos = np.argsort(-real_probs)[:n_pick]
                hard_real_indices = real_indices[hardest_pos]
                sample_weights[:] = 1.0
                sample_weights[hard_real_indices] = float(hard_neg_weight_multiplier)
                train_loader = _make_train_loader()
                print(
                    (
                        f"[SWIN][hard-negative] epoch={epoch + 1} "
                        f"selected_real={len(hard_real_indices)} "
                        f"weight_multiplier={hard_neg_weight_multiplier:.3f}"
                    ),
                    flush=True,
                )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                (
                    f"[SWIN][early_stop] epoch={epoch + 1} "
                    f"patience={early_stopping_patience} "
                    f"best_epoch={best_epoch} best_val_bal_acc={best_val_score:.6f}"
                ),
                flush=True,
            )
            break

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_threshold": float(best_threshold),
        "best_val_balanced_accuracy": float(best_val_score),
        "best_val_auc": float(best_auc),
        "best_val_real_recall": float(best_real_recall),
        "best_val_fake_recall": float(best_fake_recall),
        "selection_constraint_feasible": bool(best_feasible),
        "best_epoch": int(best_epoch),
        "epochs_ran": len(history),
        "optimizer": {
            "scheduler": scheduler_name,
            "warmup_epochs": warmup_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": max_grad_norm,
            "weight_decay": weight_decay,
            "amp": use_amp,
        },
        "advanced": {
            "selection": {
                "enabled": constrained_enabled,
                "min_fake_recall": min_fake_recall,
            },
            "hard_negative_mining": {
                "enabled": hard_neg_enabled,
                "top_k": hard_neg_top_k,
                "real_weight_multiplier": hard_neg_weight_multiplier,
                "min_epoch_to_start": hard_neg_min_epoch,
                "update_every_epochs": hard_neg_update_every,
            },
            "robust_augmentation": {
                "enabled": use_robust_aug,
            },
        },
        "history": history,
    }


def _evaluate_swin(cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    import timm

    model_cfg = cfg["models"]["swin"]
    checkpoint_path = run_dir / "model_checkpoint.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Swin checkpoint missing: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    best_threshold = float(ckpt.get("best_threshold", cfg["training"]["default_threshold"]))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_tf = transforms.Compose(
        [
            transforms.Resize((model_cfg["img_size"], model_cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_ds = ImageFolder(str(prepared_root / "val"), transform=test_tf)
    test_ds = ImageFolder(str(prepared_root / "test"), transform=test_tf)
    fake_idx = _resolve_fake_idx_from_class_to_idx(test_ds.class_to_idx, context="test")
    val_fake_idx = _resolve_fake_idx_from_class_to_idx(val_ds.class_to_idx, context="val")
    if val_fake_idx != fake_idx:
        raise RuntimeError(
            (
                "Inconsistent fake class index between val and test splits for evaluation: "
                f"val={val_fake_idx}, test={fake_idx}. "
                f"val_class_to_idx={val_ds.class_to_idx}, test_class_to_idx={test_ds.class_to_idx}"
            )
        )
    ckpt_fake_idx = ckpt.get("fake_idx", None)
    if ckpt_fake_idx is not None and int(ckpt_fake_idx) != fake_idx:
        print(
            (
                "[SWIN][eval] checkpoint fake_idx differs from dataset class mapping; "
                f"using dataset mapping fake_idx={fake_idx} "
                f"(ckpt_fake_idx={int(ckpt_fake_idx)})."
            ),
            flush=True,
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_opt = _cfg_get(cfg, ["training", "optimization", "dataloader"], {})
    num_workers = int(model_cfg["num_workers"])
    pin_memory = _resolve_flag(dataloader_opt.get("pin_memory", "auto"), auto_value=(device == "cuda"))
    persistent_workers = _resolve_flag(
        dataloader_opt.get("persistent_workers", "auto"),
        auto_value=(num_workers > 0),
    )
    prefetch_factor = max(2, int(dataloader_opt.get("prefetch_factor", 2)))
    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    val_loader = DataLoader(
        val_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )

    model = timm.create_model(model_cfg["model_name"], pretrained=False, num_classes=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_logits: List[np.ndarray] = []
    val_y_raw: List[np.ndarray] = []
    test_logits: List[np.ndarray] = []
    test_y_raw: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1).detach().cpu().numpy()
            val_logits.append(logits)
            val_y_raw.append(y.numpy())
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1).detach().cpu().numpy()
            test_logits.append(logits)
            test_y_raw.append(y.numpy())

    val_raw = np.concatenate(val_y_raw).astype(int)
    val_y_bin = _binary_label_from_class_idx(val_raw, fake_idx=fake_idx)
    val_logits_np = np.concatenate(val_logits)

    eval_adv_cfg = _cfg_get(cfg, ["training", "advanced", "evaluation"], {})
    calibration_cfg = _cfg_get(cfg, ["training", "advanced", "calibration"], {})
    constrained_cfg = _cfg_get(cfg, ["training", "advanced", "selection"], {})
    calibration_enabled = bool(calibration_cfg.get("enabled", False))
    constrained_enabled = bool(constrained_cfg.get("enabled", False))
    min_fake_recall = float(constrained_cfg.get("min_fake_recall", 0.95))

    temperature = 1.0
    if calibration_enabled:
        temperature = _fit_temperature_scaler(
            logits=val_logits_np,
            labels=val_y_bin,
            max_steps=int(calibration_cfg.get("max_steps", 200)),
            lr=float(calibration_cfg.get("lr", 0.05)),
        )

    val_probs = sigmoid(val_logits_np / float(temperature))
    if constrained_enabled:
        threshold, threshold_payload = find_threshold_max_real_recall(
            y_true=val_y_bin,
            probs_fake=val_probs,
            min_fake_recall=min_fake_recall,
        )
    else:
        threshold, _ = find_best_threshold(val_y_bin, val_probs)
        counts_tmp = confusion_counts(val_y_bin, val_probs, threshold)
        recalls_tmp = classwise_recalls_from_counts(counts_tmp)
        threshold_payload = {
            "fake_recall": float(recalls_tmp["fake_recall"]),
            "real_recall": float(recalls_tmp["real_recall"]),
            "balanced_accuracy": float(0.5 * (recalls_tmp["fake_recall"] + recalls_tmp["real_recall"])),
            "feasible": 1.0,
        }

    if bool(eval_adv_cfg.get("prefer_checkpoint_threshold", False)):
        threshold = best_threshold

    test_raw = np.concatenate(test_y_raw).astype(int)
    y_bin = _binary_label_from_class_idx(test_raw, fake_idx=fake_idx)
    probs = sigmoid(np.concatenate(test_logits) / float(temperature))
    metrics = compute_binary_metrics(y_bin, probs, threshold)
    metrics["temperature"] = float(temperature)
    metrics["val_selected_threshold"] = float(threshold)
    metrics["val_constraint_fake_recall"] = float(threshold_payload["fake_recall"])
    metrics["val_constraint_real_recall"] = float(threshold_payload["real_recall"])
    metrics["val_constraint_balanced_accuracy"] = float(threshold_payload["balanced_accuracy"])
    metrics["val_constraint_feasible"] = bool(threshold_payload["feasible"] >= 0.5)
    metrics.update(pr_auc_fake_and_real(y_bin, probs))

    samples = test_ds.samples
    predictions: List[Dict[str, Any]] = []
    for i, (path, class_idx) in enumerate(samples):
        prob = float(probs[i])
        pred = int(prob >= threshold)
        predictions.append(
            {
                "path": str(path),
                "true_label": int(class_idx == fake_idx),
                "prob_fake": prob,
                "pred_label": pred,
                "threshold": float(threshold),
            }
        )
    return {"metrics": metrics, "predictions": predictions}


class _InferenceDataset:
    def __init__(self, image_paths: Sequence[Path], transform: Any):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        import torch

        image = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(image)
        return tensor, str(self.image_paths[idx])


def _infer_swin(cfg: Dict[str, Any], run_dir: Path, input_dir: Path) -> List[Dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import timm

    model_cfg = cfg["models"]["swin"]
    checkpoint_path = run_dir / "model_checkpoint.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Swin checkpoint missing: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    threshold = float(ckpt.get("best_threshold", cfg["training"]["default_threshold"]))
    temperature = 1.0
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "val_selected_threshold" in metrics_payload:
                threshold = float(metrics_payload["val_selected_threshold"])
            if "temperature" in metrics_payload:
                temperature = float(metrics_payload["temperature"])
        except Exception:
            pass

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize((model_cfg["img_size"], model_cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    image_paths = _load_image_paths_for_inference(input_dir)
    ds = _InferenceDataset(image_paths, transform)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_opt = _cfg_get(cfg, ["training", "optimization", "dataloader"], {})
    pin_memory = _resolve_flag(dataloader_opt.get("pin_memory", "auto"), auto_value=(device == "cuda"))
    loader = DataLoader(
        ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = timm.create_model(model_cfg["model_name"], pretrained=False, num_classes=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1).detach().cpu().numpy()
            probs = sigmoid(logits / float(temperature))
            for path, prob in zip(paths, probs):
                pred = int(prob >= threshold)
                rows.append(
                    {
                        "path": str(path),
                        "prob_fake": float(prob),
                        "pred_label": pred,
                        "threshold": float(threshold),
                        "temperature": float(temperature),
                    }
                )
    return rows


def _tf_variant(variant: str):
    import tensorflow as tf

    variant = variant.upper()
    if variant == "B0":
        return tf.keras.applications.EfficientNetB0
    if variant == "B1":
        return tf.keras.applications.EfficientNetB1
    if variant == "B2":
        return tf.keras.applications.EfficientNetB2
    if variant == "B3":
        return tf.keras.applications.EfficientNetB3
    raise ValueError(f"Unsupported EfficientNet variant: {variant}")


def _build_efficientnet_model(cfg: Dict[str, Any]):
    import tensorflow as tf

    model_cfg = cfg["models"]["efficientnet"]
    backbone_cls = _tf_variant(model_cfg["variant"])
    inputs = tf.keras.Input(shape=(model_cfg["img_size"], model_cfg["img_size"], 3))
    backbone = backbone_cls(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = not bool(model_cfg["freeze_backbone"])
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(float(model_cfg["dropout"]))(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(model_cfg["lr"])),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _tf_dataset_from_directory(
    root: Path,
    cfg: Dict[str, Any],
    training: bool,
):
    import tensorflow as tf

    model_cfg = cfg["models"]["efficientnet"]
    class_names = cfg["data"]["class_names"]
    ds = tf.keras.utils.image_dataset_from_directory(
        str(root),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(model_cfg["img_size"], model_cfg["img_size"]),
        batch_size=model_cfg["batch_size"],
        shuffle=training,
        seed=int(cfg["project"]["seed"]),
    )
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def _train_efficientnet(cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    import tensorflow as tf

    model_cfg = cfg["models"]["efficientnet"]
    fake_idx = cfg["data"]["class_names"].index("fake")
    if fake_idx != 1:
        raise RuntimeError("Config class_names must map fake to index 1 for binary label semantics.")

    train_ds = _tf_dataset_from_directory(prepared_root / "train", cfg, training=True)
    val_ds = _tf_dataset_from_directory(prepared_root / "val", cfg, training=False)
    model = _build_efficientnet_model(cfg)
    eff_opt = _cfg_get(cfg, ["training", "optimization", "efficientnet"], {})

    callbacks: List[Any] = []
    early_stopping_patience = max(0, int(eff_opt.get("early_stopping_patience", 0)))
    early_stopping_min_delta = float(eff_opt.get("early_stopping_min_delta", 0.0))
    if early_stopping_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=True,
            )
        )

    reduce_lr_patience = max(0, int(eff_opt.get("reduce_lr_patience", 0)))
    if reduce_lr_patience > 0:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                mode="max",
                factor=float(eff_opt.get("reduce_lr_factor", 0.5)),
                patience=reduce_lr_patience,
                min_lr=float(eff_opt.get("min_lr", 1e-6)),
                verbose=1,
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(model_cfg["epochs"]),
        verbose=1,
        callbacks=callbacks or None,
    )

    logits = model.predict(val_ds, verbose=0).reshape(-1)
    labels = np.concatenate([y.numpy() for _, y in val_ds]).astype(int)
    probs = sigmoid(logits)
    threshold, score = find_best_threshold(labels, probs)
    metrics = compute_binary_metrics(labels, probs, threshold)

    checkpoint_path = run_dir / "model_checkpoint.keras"
    model.save(checkpoint_path)
    metadata_path = run_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "best_threshold": float(threshold),
                "best_val_balanced_accuracy": float(score),
                "best_val_auc": float(metrics["roc_auc"]),
                "history_keys": list(history.history.keys()),
                "epochs_ran": len(history.history.get("loss", [])),
                "callbacks": {
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "reduce_lr_patience": reduce_lr_patience,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "best_threshold": float(threshold),
        "best_val_balanced_accuracy": float(score),
        "best_val_auc": float(metrics["roc_auc"]),
    }


def _evaluate_efficientnet(cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    import tensorflow as tf

    checkpoint_path = run_dir / "model_checkpoint.keras"
    metadata_path = run_dir / "model_metadata.json"
    if not checkpoint_path.exists():
        raise RuntimeError(f"EfficientNet checkpoint missing: {checkpoint_path}")
    if not metadata_path.exists():
        raise RuntimeError(f"EfficientNet metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata.get("best_threshold", cfg["training"]["default_threshold"]))

    test_ds = _tf_dataset_from_directory(prepared_root / "test", cfg, training=False)
    model = tf.keras.models.load_model(checkpoint_path)
    logits = model.predict(test_ds, verbose=0).reshape(-1)
    labels = np.concatenate([y.numpy() for _, y in test_ds]).astype(int)
    probs = sigmoid(logits)
    metrics = compute_binary_metrics(labels, probs, threshold)

    # Rebuild path order from directory listing because tf dataset does not retain paths in output tensors.
    class_names = cfg["data"]["class_names"]
    image_paths: List[Path] = []
    true_labels: List[int] = []
    for class_name in class_names:
        class_dir = prepared_root / "test" / class_name
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file():
                image_paths.append(image_path)
                true_labels.append(1 if class_name == "fake" else 0)
    count = min(len(image_paths), len(probs))
    predictions = []
    for i in range(count):
        prob = float(probs[i])
        pred = int(prob >= threshold)
        predictions.append(
            {
                "path": str(image_paths[i]),
                "true_label": int(true_labels[i]),
                "prob_fake": prob,
                "pred_label": pred,
                "threshold": float(threshold),
            }
        )
    return {"metrics": metrics, "predictions": predictions}


def _infer_efficientnet(cfg: Dict[str, Any], run_dir: Path, input_dir: Path) -> List[Dict[str, Any]]:
    import tensorflow as tf

    checkpoint_path = run_dir / "model_checkpoint.keras"
    metadata_path = run_dir / "model_metadata.json"
    if not checkpoint_path.exists():
        raise RuntimeError(f"EfficientNet checkpoint missing: {checkpoint_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata.get("best_threshold", cfg["training"]["default_threshold"]))

    image_paths = _load_image_paths_for_inference(input_dir)
    img_size = int(cfg["models"]["efficientnet"]["img_size"])
    model = tf.keras.models.load_model(checkpoint_path)

    rows: List[Dict[str, Any]] = []
    batch: List[np.ndarray] = []
    batch_paths: List[Path] = []
    batch_size = int(cfg["models"]["efficientnet"]["batch_size"])

    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((img_size, img_size))
        batch.append(np.asarray(img, dtype=np.float32) / 255.0)
        batch_paths.append(path)
        if len(batch) >= batch_size:
            logits = model.predict(np.stack(batch), verbose=0).reshape(-1)
            probs = sigmoid(logits)
            for local_path, prob in zip(batch_paths, probs):
                rows.append(
                    {
                        "path": str(local_path),
                        "prob_fake": float(prob),
                        "pred_label": int(prob >= threshold),
                        "threshold": float(threshold),
                    }
                )
            batch = []
            batch_paths = []

    if batch:
        logits = model.predict(np.stack(batch), verbose=0).reshape(-1)
        probs = sigmoid(logits)
        for local_path, prob in zip(batch_paths, probs):
            rows.append(
                {
                    "path": str(local_path),
                    "prob_fake": float(prob),
                    "pred_label": int(prob >= threshold),
                    "threshold": float(threshold),
                }
            )
    return rows
