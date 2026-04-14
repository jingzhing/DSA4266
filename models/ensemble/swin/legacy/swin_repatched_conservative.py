import os
import math
import json
import csv
import copy
import random
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import swin_t, Swin_T_Weights

warnings.filterwarnings("ignore")

SEED = 42

IMG_SIZE = 224
EVAL_RESIZE = 232

BATCH_SIZE = 48
EPOCHS = 8
EARLY_STOP_PAT = 5
VAL_SIZE = 0.2

NUM_WORKERS = 4
PREFETCH_FACTOR = 2

BASE_LR = 1e-5
HEAD_LR = 5e-5
WEIGHT_DECAY = 0.02
MIN_LR = 1e-6
WARMUP_EPOCHS = 2

DROPOUT = 0.10
GRAD_CLIP_NORM = 1.0

USE_EMA = False
EMA_DECAY = 0.9997

USE_POS_WEIGHT = False
LABEL_SMOOTHING = 0.0

MIXUP_ALPHA = 0.0
CUTMIX_ALPHA = 0.0
MIX_PROB = 0.0

USE_FIXED_TEST_THRESHOLD = True
FIXED_THRESHOLD = 0.50
THRESHOLD_GRID = np.round(np.arange(0.20, 0.81, 0.01), 2)

CHECKPOINT_DIR = Path("models/swin/repatched_conservative_run")
CHECKPOINT_NAME = "best_swin_t_repatched_conservative.pt"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")

COMMON_ROOTS = [
    Path("data/deepdetect-2025_dddata"),
    Path("data/deepdetect-2025"),
    Path("data/ddata"),
    Path("deepdetect-2025_dddata"),
    Path("deepdetect-2025"),
    Path("ddata"),
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = False if os.name == "nt" else NUM_WORKERS > 0


def log(msg=""):
    print(msg, flush=True)


class DeepfakeDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            log(f"[WARN] Failed to read image: {path} | {e}")
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return img, torch.tensor(label, dtype=torch.float32), str(path)


class BCEWithLogitsLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.loss(logits, targets).mean()


class ModelEMA:
    def __init__(self, model, decay=0.9997, device=None):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.device = device
        for p in self.module.parameters():
            p.requires_grad_(False)
        if self.device is not None:
            self.module.to(self.device)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            if k not in model_state:
                continue
            model_v = model_state[k].detach()
            if self.device is not None:
                model_v = model_v.to(self.device)
            if not torch.is_floating_point(v):
                v.copy_(model_v)
            else:
                v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lrs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = base_lrs
        self.min_lr = min_lr

    def step(self, epoch):
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch <= self.warmup_epochs:
                lr = base_lr * epoch / max(self.warmup_epochs, 1)
            else:
                progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            group["lr"] = lr


def get_image_paths(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([str(p) for p in directory.rglob("*") if p.suffix.lower() in VALID_EXT])


def preview_dir(directory, max_items=8):
    directory = Path(directory)
    log(f"[PATH CHECK] {directory} | exists={directory.exists()} | is_dir={directory.is_dir()}")
    if not directory.exists() or not directory.is_dir():
        return
    try:
        items = list(directory.iterdir())
        log(f"[PATH CHECK] immediate items={len(items)}")
        for item in items[:max_items]:
            kind = "DIR " if item.is_dir() else "FILE"
            log(f"    - {kind} {item.name}")
    except Exception as e:
        log(f"[PATH CHECK] failed to inspect {directory}: {e}")


def find_class_dir(root_dir, split_name, class_name):
    root_dir = Path(root_dir)
    split_lower = split_name.lower()
    class_lower = class_name.lower()

    direct = root_dir / split_name / class_name
    if direct.exists() and direct.is_dir():
        return direct

    split_candidates = []
    for p in root_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == split_lower:
            split_candidates.append(p)

    for split_dir in split_candidates:
        candidate = split_dir / class_name
        if candidate.exists() and candidate.is_dir():
            return candidate
        for child in split_dir.iterdir():
            if child.is_dir() and child.name.lower() == class_lower:
                return child

    for p in root_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == class_lower and p.parent.name.lower() == split_lower:
            return p

    return direct


def detect_root():
    env_root = os.environ.get("DEEPNET_ROOT")
    if env_root:
        root = Path(env_root)
        if root.exists():
            return root

    for root in COMMON_ROOTS:
        if root.exists():
            return root

    cwd = Path.cwd()
    for candidate in cwd.rglob("train"):
        parent = candidate.parent
        if (parent / "train" / "real").exists() and (parent / "train" / "fake").exists():
            if (parent / "test" / "real").exists() and (parent / "test" / "fake").exists():
                return parent

    raise FileNotFoundError(
        "Could not find dataset root. Set DEEPNET_ROOT or place your data in one of these structures."
    )


def count_labels(labels):
    labels = np.array(labels)
    real_count = int((labels == 0).sum())
    fake_count = int((labels == 1).sum())
    return real_count, fake_count


def search_best_threshold(y_true, y_prob, grid=THRESHOLD_GRID):
    best_threshold = 0.5
    best_bal_acc = -1.0
    best_acc = -1.0

    for thr in grid:
        y_pred = (y_prob >= thr).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        if bal_acc > best_bal_acc or (bal_acc == best_bal_acc and acc > best_acc):
            best_bal_acc = bal_acc
            best_acc = acc
            best_threshold = float(thr)

    return best_threshold, best_bal_acc, best_acc


def make_loader(paths, labels, transform, shuffle, loader_name):
    ds = DeepfakeDataset(paths, labels, transform)
    loader_kwargs = dict(
        dataset=ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    loader = DataLoader(**loader_kwargs)
    log(f"[LOADER] {loader_name:<10} batches={len(loader)} samples={len(ds)} shuffle={shuffle}")
    return loader


def build_model(weights):
    model = swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 1),
    )
    return model


def apply_trainable_layers(model, epoch):
    feature_blocks = list(model.features)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    if epoch <= 3:
        trainable_mode = "head_only"
    elif epoch <= 7:
        trainable_mode = "last_stage_plus_head"
        for p in feature_blocks[-1].parameters():
            p.requires_grad = True
    else:
        trainable_mode = "last_2_stages_plus_head"
        for block in feature_blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True

    return trainable_mode


def build_optimizer(model):
    head_params = list(model.head.parameters())
    body_params = [p for name, p in model.named_parameters() if not name.startswith("head")]
    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": BASE_LR, "name": "body"},
            {"params": head_params, "lr": HEAD_LR, "name": "head"},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS,
        base_lrs=[BASE_LR, HEAD_LR],
        min_lr=MIN_LR,
    )
    return optimizer, scheduler


def run_epoch(model, loader, criterion, optimizer, scaler, threshold, train=True, epoch_num=None, stage_name="VAL", ema=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_probs = []
    all_true = []
    all_paths = []

    desc = f"Epoch {epoch_num:02d} {stage_name}" if epoch_num is not None else stage_name
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        pbar = tqdm(loader, desc=desc, leave=False, ncols=180)
        for batch_idx, batch in enumerate(pbar, start=1):
            imgs, labels, paths = batch
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=USE_AMP):
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            total_correct += (preds == labels.long()).sum().item()
            total_seen += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            all_probs.extend(probs.detach().cpu().numpy())
            all_true.extend(labels.detach().cpu().numpy())
            all_paths.extend(paths)

            running_loss = total_loss / max(total_seen, 1)
            running_acc = total_correct / max(total_seen, 1)
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                prefix = f"[{stage_name}] "
                epoch_part = f"epoch={epoch_num:02d} " if epoch_num is not None else ""
                log(
                    f"{prefix}{epoch_part}batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={running_loss:.4f} acc={running_acc:.4f}"
                )

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    y_true = np.array(all_true).astype(int)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return avg_loss, avg_acc, bal_acc, y_true, y_prob, y_pred, all_paths


def save_curves(output_dir, history, cm, fpr, tpr, roc_auc, precision, recall, ap, y_true, y_prob, threshold):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["val_bal_acc"], label="Val balanced acc")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced accuracy")
    plt.title("Validation balanced accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="Real")
    plt.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="Fake")
    plt.axvline(threshold, linestyle="--", linewidth=1.2, label=f"Threshold = {threshold:.2f}")
    plt.xlabel("Predicted P(Fake)")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_prediction_csv(output_path, paths, y_true, y_prob, y_pred):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "prob_fake", "pred_label"])
        for p, yt, yp, yd in zip(paths, y_true, y_prob, y_pred):
            writer.writerow([p, int(yt), float(yp), int(yd)])


def main():
    ROOT_DIR = detect_root()
    TRAIN_REAL_DIR = find_class_dir(ROOT_DIR, "train", "real")
    TRAIN_FAKE_DIR = find_class_dir(ROOT_DIR, "train", "fake")
    TEST_REAL_DIR = find_class_dir(ROOT_DIR, "test", "real")
    TEST_FAKE_DIR = find_class_dir(ROOT_DIR, "test", "fake")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR = CHECKPOINT_DIR / "artifacts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("SWIN-T REPATCHED CONSERVATIVE RUN")
    log("=" * 80)
    log(f"Device               : {DEVICE}")
    log(f"AMP                  : {USE_AMP}")
    log(f"Dataset root         : {ROOT_DIR.resolve()}")
    log(f"Train real dir       : {TRAIN_REAL_DIR}")
    log(f"Train fake dir       : {TRAIN_FAKE_DIR}")
    log(f"Test real dir        : {TEST_REAL_DIR}")
    log(f"Test fake dir        : {TEST_FAKE_DIR}")
    log(f"Batch size           : {BATCH_SIZE}")
    log(f"Epochs               : {EPOCHS}")
    log(f"Image size           : {IMG_SIZE}")
    log(f"Workers              : {NUM_WORKERS}")
    log(f"Persistent workers   : {PERSISTENT_WORKERS}")
    log(f"Prefetch factor      : {PREFETCH_FACTOR if NUM_WORKERS > 0 else 'n/a'}")
    log(f"Base learning rate   : {BASE_LR}")
    log(f"Head learning rate   : {HEAD_LR}")
    log(f"Weight decay         : {WEIGHT_DECAY}")
    log(f"Dropout              : {DROPOUT}")
    log(f"EMA enabled          : {USE_EMA}")
    log(f"Label smoothing      : {LABEL_SMOOTHING}")
    log(f"Mix probability      : {MIX_PROB}")
    log(f"Fixed threshold      : {FIXED_THRESHOLD if USE_FIXED_TEST_THRESHOLD else 'search_on_val'}")
    log(f"Checkpoint           : {CHECKPOINT_DIR / CHECKPOINT_NAME}")

    log("\n[DIRECTORY PREVIEW]")
    preview_dir(ROOT_DIR)
    preview_dir(ROOT_DIR / "train")
    preview_dir(ROOT_DIR / "test")
    preview_dir(TRAIN_REAL_DIR)
    preview_dir(TRAIN_FAKE_DIR)
    preview_dir(TEST_REAL_DIR)
    preview_dir(TEST_FAKE_DIR)

    train_real_paths = get_image_paths(TRAIN_REAL_DIR)
    train_fake_paths = get_image_paths(TRAIN_FAKE_DIR)
    test_real_paths = get_image_paths(TEST_REAL_DIR)
    test_fake_paths = get_image_paths(TEST_FAKE_DIR)

    log("\n[IMAGE COUNTS]")
    log(f"Train real images    : {len(train_real_paths)}")
    log(f"Train fake images    : {len(train_fake_paths)}")
    log(f"Test real images     : {len(test_real_paths)}")
    log(f"Test fake images     : {len(test_fake_paths)}")

    if len(train_real_paths) == 0 or len(train_fake_paths) == 0:
        raise RuntimeError("Training folders are empty or not found.")
    if len(test_real_paths) == 0 or len(test_fake_paths) == 0:
        raise RuntimeError("Test folders are empty or not found.")

    train_paths = train_real_paths + train_fake_paths
    train_labels = [0] * len(train_real_paths) + [1] * len(train_fake_paths)
    test_paths = test_real_paths + test_fake_paths
    test_labels = [0] * len(test_real_paths) + [1] * len(test_fake_paths)

    TR_PATHS, VA_PATHS, TR_LABELS, VA_LABELS = train_test_split(
        train_paths,
        train_labels,
        test_size=VAL_SIZE,
        stratify=train_labels,
        random_state=SEED,
    )

    tr_real, tr_fake = count_labels(TR_LABELS)
    va_real, va_fake = count_labels(VA_LABELS)
    te_real, te_fake = count_labels(test_labels)

    log("\n[DATA SPLIT]")
    log(f"Train samples        : {len(TR_PATHS)} | real={tr_real} fake={tr_fake}")
    log(f"Val samples          : {len(VA_PATHS)} | real={va_real} fake={va_fake}")
    log(f"Test samples         : {len(test_paths)} | real={te_real} fake={te_fake}")

    with open(ARTIFACT_DIR / "split_debug.txt", "w", encoding="utf-8") as f:
        f.write("TRAIN\n")
        for p, y in zip(TR_PATHS[:500], TR_LABELS[:500]):
            f.write(f"{y}\t{p}\n")
        f.write("\nVAL\n")
        for p, y in zip(VA_PATHS[:500], VA_LABELS[:500]):
            f.write(f"{y}\t{p}\n")
        f.write("\nTEST\n")
        for p, y in zip(test_paths[:500], test_labels[:500]):
            f.write(f"{y}\t{p}\n")

    weights = Swin_T_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.Resize(EVAL_RESIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(EVAL_RESIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_loader = make_loader(TR_PATHS, TR_LABELS, train_transform, True, "train")
    val_loader = make_loader(VA_PATHS, VA_LABELS, eval_transform, False, "val")
    test_loader = make_loader(test_paths, test_labels, eval_transform, False, "test")

    model = build_model(weights).to(DEVICE)
    apply_trainable_layers(model, 1)
    log(f"Model device         : {next(model.parameters()).device}")

    pos_weight = None
    pos_weight_value = None
    if USE_POS_WEIGHT:
        num_pos = sum(TR_LABELS)
        num_neg = len(TR_LABELS) - num_pos
        pos_weight_value = num_neg / max(num_pos, 1)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)

    criterion = BCEWithLogitsLabelSmoothing(smoothing=LABEL_SMOOTHING, pos_weight=pos_weight)
    scaler = GradScaler("cuda", enabled=USE_AMP)
    optimizer, scheduler = build_optimizer(model)
    ema = ModelEMA(model, decay=EMA_DECAY, device=DEVICE) if USE_EMA else None

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "val_bal_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1
    best_threshold = FIXED_THRESHOLD if USE_FIXED_TEST_THRESHOLD else 0.5
    patience_count = 0
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME

    log("\n[TRAINING START]")
    for epoch in range(1, EPOCHS + 1):
        trainable_mode = apply_trainable_layers(model, epoch)
        scheduler.step(epoch)
        current_lrs = [group["lr"] for group in optimizer.param_groups]

        log("-" * 80)
        log(f"Epoch {epoch:02d}/{EPOCHS} | mode={trainable_mode} | lrs={[f'{x:.7f}' for x in current_lrs]}")

        tr_loss, tr_acc, tr_bal_acc, _, _, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            threshold=0.5,
            train=True,
            epoch_num=epoch,
            stage_name="TRAIN",
            ema=ema,
        )

        eval_model = ema.module if (USE_EMA and ema is not None) else model
        va_loss, _, _, va_true, va_prob, _, va_paths = run_epoch(
            model=eval_model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            threshold=0.5,
            train=False,
            epoch_num=epoch,
            stage_name="VAL",
            ema=None,
        )

        if USE_FIXED_TEST_THRESHOLD:
            val_threshold = FIXED_THRESHOLD
            val_pred = (va_prob >= val_threshold).astype(int)
            val_bal_acc = balanced_accuracy_score(va_true, val_pred)
            val_acc = accuracy_score(va_true, val_pred)
        else:
            val_threshold, val_bal_acc, val_acc = search_best_threshold(va_true, va_prob)

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)

        log(
            f"[EPOCH SUMMARY] {epoch:02d}/{EPOCHS} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_bal_acc={tr_bal_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={val_acc:.4f} val_bal_acc={val_bal_acc:.4f} "
            f"val_best_thr={val_threshold:.2f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            best_threshold = val_threshold
            patience_count = 0
            save_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.module.state_dict() if (USE_EMA and ema is not None) else None,
                "best_val_loss": best_val_loss,
                "best_threshold": best_threshold,
                "history": history,
                "config": {
                    "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "base_lr": BASE_LR,
                    "head_lr": HEAD_LR,
                    "dropout": DROPOUT,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                    "root_dir": str(ROOT_DIR),
                    "num_workers": NUM_WORKERS,
                    "persistent_workers": PERSISTENT_WORKERS,
                    "use_ema": USE_EMA,
                    "ema_decay": EMA_DECAY,
                    "use_pos_weight": USE_POS_WEIGHT,
                    "pos_weight_value": pos_weight_value,
                    "label_smoothing": LABEL_SMOOTHING,
                    "mixup_alpha": MIXUP_ALPHA,
                    "cutmix_alpha": CUTMIX_ALPHA,
                    "mix_prob": MIX_PROB,
                    "use_fixed_test_threshold": USE_FIXED_TEST_THRESHOLD,
                    "fixed_threshold": FIXED_THRESHOLD,
                },
            }
            torch.save(save_state, checkpoint_path)
            save_prediction_csv(
                ARTIFACT_DIR / "best_val_predictions.csv",
                va_paths,
                va_true,
                va_prob,
                (va_prob >= val_threshold).astype(int),
            )
            log(f"[CHECKPOINT] Saved new best model at epoch {epoch:02d} -> {checkpoint_path}")
        else:
            patience_count += 1
            log(f"[EARLY STOP WATCH] No improvement count = {patience_count}/{EARLY_STOP_PAT}")
            if patience_count >= EARLY_STOP_PAT:
                log(f"[EARLY STOP] Stopping at epoch {epoch:02d}")
                break

    if not checkpoint_path.exists():
        raise RuntimeError("No checkpoint was saved. Training may have failed early.")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["ema_state_dict"] if (USE_EMA and ckpt.get("ema_state_dict") is not None) else ckpt["model_state_dict"])
    best_threshold = float(ckpt.get("best_threshold", FIXED_THRESHOLD if USE_FIXED_TEST_THRESHOLD else 0.5))
    best_epoch = int(ckpt["epoch"])

    log("\n[BEST MODEL RESTORED]")
    log(f"Best epoch           : {best_epoch}")
    log(f"Best val loss        : {ckpt['best_val_loss']:.4f}")
    log(f"Best val threshold   : {best_threshold:.2f}")

    log("\n[TEST EVALUATION]")
    test_loss, _, _, y_true, y_prob, _, test_eval_paths = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        threshold=best_threshold,
        train=False,
        epoch_num=None,
        stage_name="TEST",
        ema=None,
    )

    y_pred = (y_prob >= best_threshold).astype(int)
    test_acc = accuracy_score(y_true, y_pred)
    test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    log(f"Test loss            : {test_loss:.4f}")
    log(f"Test acc             : {test_acc:.4f}")
    log(f"Test balanced acc    : {test_bal_acc:.4f}")
    log(f"Test threshold       : {best_threshold:.2f}")
    log(f"ROC AUC              : {roc_auc:.4f}")
    log(f"Average precision    : {ap:.4f}")
    log("\n[CLASSIFICATION REPORT]")
    log(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))
    log("[CONFUSION MATRIX]")
    log(str(cm))

    save_prediction_csv(ARTIFACT_DIR / "test_predictions.csv", test_eval_paths, y_true, y_prob, y_pred)

    save_curves(
        output_dir=ARTIFACT_DIR,
        history=history,
        cm=cm,
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        ap=ap,
        y_true=y_true,
        y_prob=y_prob,
        threshold=best_threshold,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": float(ckpt["best_val_loss"]),
        "best_threshold": float(best_threshold),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_bal_acc": float(test_bal_acc),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "confusion_matrix": cm.tolist(),
    }

    with open(ARTIFACT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(ARTIFACT_DIR / "metrics_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")

    log("\n[RUN COMPLETE]")
    log(f"Checkpoint saved at  : {checkpoint_path.resolve()}")
    log(f"Artifacts saved at   : {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
