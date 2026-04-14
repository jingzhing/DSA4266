
import os
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

warnings.filterwarnings("ignore")

SEED = 42
IMG_SIZE = 224
TRAIN_RESIZE = 236
EVAL_RESIZE = 236
BATCH_SIZE = 128
EPOCHS = 12
EARLY_STOP_PAT = 2
NUM_WORKERS = 4
VAL_SIZE = 0.2

BACKBONE_LR = 1e-4
HEAD_LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25

RANDOM_ERASE_PROB = 0.05

USE_LIGHT_MIXUP = True
MIXUP_ALPHA = 0.15
MIXUP_PROB = 0.15

USE_EMA = False
LABEL_SMOOTHING = 0.0

UNFREEZE_EPOCH = 4
WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.05

THRESHOLD_DEFAULT = 0.5
THRESHOLD_SEARCH_MIN = 0.20
THRESHOLD_SEARCH_MAX = 0.80
THRESHOLD_SEARCH_STEP = 0.01

CHECKPOINT_DIR = Path("models/efficientnet/balanced_v4_run")
CHECKPOINT_NAME = "best_efficientnet_b0_balanced_v4.pt"
ARTIFACTS_DIRNAME = "artifacts"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")

COMMON_ROOTS = [
    Path("data/deepdetect-2025_dddata"),
    Path("data/deepdetect-2025"),
    Path("data/ddata"),
    Path("deepdetect-2025_dddata"),
    Path("deepdetect-2025"),
    Path("ddata"),
]

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def log(msg=""):
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepfakeDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception as exc:
            log(f"[WARN] Failed to read image: {path} | {exc}")
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return img, torch.tensor(label, dtype=torch.float32), str(path)


class LightMixupCollator:
    def __init__(self, alpha=0.15, prob=0.15):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.stack([b[1] for b in batch], dim=0)
        paths = [b[2] for b in batch]

        if len(batch) < 2 or np.random.rand() > self.prob:
            return imgs, labels, labels, 1.0, paths

        indices = torch.randperm(imgs.size(0))
        lam = np.random.beta(self.alpha, self.alpha)
        lam = float(max(lam, 1.0 - lam))
        mixed_imgs = lam * imgs + (1.0 - lam) * imgs[indices]
        return mixed_imgs, labels, labels[indices], lam, paths


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.05):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(total_epochs, self.warmup_epochs + 1)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch_idx):
        epoch = epoch_idx + 1
        if epoch <= self.warmup_epochs:
            mult = epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            mult = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * mult


def get_image_paths(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(str(p) for p in directory.rglob("*") if p.suffix.lower() in VALID_EXT)


def preview_dir(directory, max_items=8):
    directory = Path(directory)
    log(f"[PATH CHECK] {directory} | exists={directory.exists()} | is_dir={directory.is_dir()}")
    if not directory.exists() or not directory.is_dir():
        return
    items = list(directory.iterdir())
    log(f"[PATH CHECK] immediate items={len(items)}")
    for item in items[:max_items]:
        kind = "DIR " if item.is_dir() else "FILE"
        log(f"    - {kind} {item.name}")


def find_class_dir(root_dir, split_name, class_name):
    root_dir = Path(root_dir)
    direct = root_dir / split_name / class_name
    if direct.exists() and direct.is_dir():
        return direct

    split_lower = split_name.lower()
    class_lower = class_name.lower()

    for p in root_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == split_lower:
            candidate = p / class_name
            if candidate.exists() and candidate.is_dir():
                return candidate
            for child in p.iterdir():
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
        train_real = find_class_dir(parent, "train", "real")
        train_fake = find_class_dir(parent, "train", "fake")
        test_real = find_class_dir(parent, "test", "real")
        test_fake = find_class_dir(parent, "test", "fake")
        if train_real.exists() and train_fake.exists() and test_real.exists() and test_fake.exists():
            return parent

    raise FileNotFoundError(
        "Could not find dataset root. Set DEEPNET_ROOT or place your data in one of the supported folder layouts."
    )


def count_labels(labels):
    labels = np.asarray(labels)
    return int((labels == 0).sum()), int((labels == 1).sum())


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((TRAIN_RESIZE, TRAIN_RESIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.90, 1.0), ratio=(0.95, 1.05), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.04, contrast=0.04, saturation=0.04, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
        transforms.RandomErasing(p=RANDOM_ERASE_PROB, scale=(0.02, 0.06), ratio=(0.7, 1.4), value="random"),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((EVAL_RESIZE, EVAL_RESIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    return train_transform, eval_transform


def make_loader(paths, labels, transform, shuffle, loader_name, collate_fn=None):
    ds = DeepfakeDataset(paths, labels, transform)
    kwargs = dict(
        dataset=ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    loader = DataLoader(**kwargs)
    log(f"[LOADER] {loader_name:<10} batches={len(loader)} samples={len(ds)} shuffle={shuffle}")
    return loader


def build_model(dropout=0.25):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for block in model.features[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def maybe_unfreeze_all(model, optimizer, current_epoch, unfreeze_epoch=4):
    if current_epoch != unfreeze_epoch:
        return optimizer

    log("[FINE-TUNE] Unfreezing full backbone from this epoch onward")
    for param in model.features.parameters():
        param.requires_grad = True

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    new_optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    try:
        new_optimizer.load_state_dict(optimizer.state_dict())
        new_optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]
        new_optimizer.param_groups[1]["lr"] = optimizer.param_groups[1]["lr"]
    except Exception:
        pass
    return new_optimizer


def run_epoch(loader, model, criterion, optimizer, scaler, device, train=True, epoch_num=None, stage_name=None, threshold=0.5):
    stage_name = stage_name or ("TRAIN" if train else "EVAL")
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_probs = []
    all_true = []
    all_paths = []

    desc = f"Epoch {epoch_num:02d} {stage_name}" if epoch_num is not None else stage_name
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(pbar, start=1):
            if train:
                imgs, labels_a, labels_b, lam, paths = batch
                labels_a = labels_a.to(device, non_blocking=True)
                labels_b = labels_b.to(device, non_blocking=True)
                labels = labels_a
            else:
                imgs, labels, paths = batch
                labels = labels.to(device, non_blocking=True)
                labels_a, labels_b, lam = labels, labels, 1.0

            imgs = imgs.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(imgs).squeeze(1)
                if train and lam < 1.0:
                    loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            batch_correct = (preds == labels.long()).sum().item()

            total_correct += batch_correct
            total_seen += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            all_probs.extend(probs.detach().cpu().numpy())
            all_true.extend(labels.detach().cpu().numpy())
            all_paths.extend(paths)

            running_loss = total_loss / max(total_seen, 1)
            running_acc = total_correct / max(total_seen, 1)
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                epoch_part = f"epoch={epoch_num:02d} " if epoch_num is not None else ""
                log(
                    f"[{stage_name}] {epoch_part}batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={running_loss:.4f} acc={running_acc:.4f}"
                )

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    y_true = np.asarray(all_true).astype(int)
    y_prob = np.asarray(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return avg_loss, avg_acc, bal_acc, y_true, y_prob, y_pred, all_paths


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(THRESHOLD_SEARCH_MIN, THRESHOLD_SEARCH_MAX + 1e-9, THRESHOLD_SEARCH_STEP)
    best_threshold = THRESHOLD_DEFAULT
    best_score = -1.0
    records = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = (y_pred == y_true).mean()
        records.append((thr, bal_acc, acc))
        if bal_acc > best_score:
            best_score = bal_acc
            best_threshold = float(thr)

    return best_threshold, best_score, records


def evaluate_thresholds(y_true, y_prob, thresholds):
    out = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        out.append({
            "threshold": thr,
            "accuracy": float((y_pred == y_true).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "cm": confusion_matrix(y_true, y_pred).tolist(),
        })
    return out


def save_artifacts(output_dir, history, y_true, y_prob, y_pred, threshold, report_text):
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    with open(output_dir / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"threshold={threshold:.4f}\n")
        f.write(f"accuracy={(y_pred == y_true).mean():.6f}\n")
        f.write(f"balanced_accuracy={balanced_accuracy_score(y_true, y_pred):.6f}\n")
        f.write(f"roc_auc={roc_auc:.6f}\n")
        f.write(f"average_precision={ap:.6f}\n\n")
        f.write("classification_report\n")
        f.write(report_text)
        f.write("\nconfusion_matrix\n")
        f.write(str(cm))
        f.write("\n")

    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=200)
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
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="Real")
    plt.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="Fake")
    plt.axvline(threshold, linestyle="--", linewidth=1.2, label=f"Threshold = {threshold:.2f}")
    plt.xlabel("Predicted P(Fake)")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=200)
    plt.close()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    root_dir = detect_root()
    train_real_dir = find_class_dir(root_dir, "train", "real")
    train_fake_dir = find_class_dir(root_dir, "train", "fake")
    test_real_dir = find_class_dir(root_dir, "test", "real")
    test_fake_dir = find_class_dir(root_dir, "test", "fake")

    log("=" * 80)
    log("EFFICIENTNET B0 BALANCED V4 TRAINING RUN")
    log("=" * 80)
    log(f"Device               : {device}")
    log(f"AMP                  : {use_amp}")
    log(f"Dataset root         : {root_dir.resolve()}")
    log(f"Train real dir       : {train_real_dir}")
    log(f"Train fake dir       : {train_fake_dir}")
    log(f"Test real dir        : {test_real_dir}")
    log(f"Test fake dir        : {test_fake_dir}")
    log(f"Batch size           : {BATCH_SIZE}")
    log(f"Epochs               : {EPOCHS}")
    log(f"Image size           : {IMG_SIZE}")
    log(f"Workers              : {NUM_WORKERS}")
    log(f"Checkpoint           : {CHECKPOINT_DIR / CHECKPOINT_NAME}")

    log("\n[DIRECTORY PREVIEW]")
    preview_dir(root_dir)
    preview_dir(root_dir / "train")
    preview_dir(root_dir / "test")
    preview_dir(train_real_dir)
    preview_dir(train_fake_dir)
    preview_dir(test_real_dir)
    preview_dir(test_fake_dir)

    train_real_paths = get_image_paths(train_real_dir)
    train_fake_paths = get_image_paths(train_fake_dir)
    test_real_paths = get_image_paths(test_real_dir)
    test_fake_paths = get_image_paths(test_fake_dir)

    log("\n[IMAGE COUNTS]")
    log(f"Train real images    : {len(train_real_paths)}")
    log(f"Train fake images    : {len(train_fake_paths)}")
    log(f"Test real images     : {len(test_real_paths)}")
    log(f"Test fake images     : {len(test_fake_paths)}")

    if not train_real_paths or not train_fake_paths:
        raise RuntimeError("Training folders are empty or not found.")
    if not test_real_paths or not test_fake_paths:
        raise RuntimeError("Test folders are empty or not found.")

    train_paths = train_real_paths + train_fake_paths
    train_labels = [0] * len(train_real_paths) + [1] * len(train_fake_paths)
    test_paths = test_real_paths + test_fake_paths
    test_labels = [0] * len(test_real_paths) + [1] * len(test_fake_paths)

    tr_paths, va_paths, tr_labels, va_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=VAL_SIZE,
        stratify=train_labels,
        random_state=SEED,
    )

    tr_real, tr_fake = count_labels(tr_labels)
    va_real, va_fake = count_labels(va_labels)
    te_real, te_fake = count_labels(test_labels)

    log("\n[DATA SPLIT]")
    log(f"Train samples        : {len(tr_paths)} | real={tr_real} fake={tr_fake}")
    log(f"Val samples          : {len(va_paths)} | real={va_real} fake={va_fake}")
    log(f"Test samples         : {len(test_paths)} | real={te_real} fake={te_fake}")
    log("\n[DEBUG SAMPLE PATHS]")
    for i, p in enumerate(tr_paths[:3]):
        log(f"Train sample {i + 1}      : {p}")
    for i, p in enumerate(va_paths[:3]):
        log(f"Val sample {i + 1}        : {p}")
    for i, p in enumerate(test_paths[:3]):
        log(f"Test sample {i + 1}       : {p}")

    train_transform, eval_transform = build_transforms()
    train_collate = LightMixupCollator(alpha=MIXUP_ALPHA, prob=MIXUP_PROB) if USE_LIGHT_MIXUP else None
    train_loader = make_loader(tr_paths, tr_labels, train_transform, True, "train", collate_fn=train_collate)
    val_loader = make_loader(va_paths, va_labels, eval_transform, False, "val")
    test_loader = make_loader(test_paths, test_labels, eval_transform, False, "test")

    model = build_model(dropout=DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss()

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS, min_lr_ratio=MIN_LR_RATIO)
    scaler = GradScaler("cuda", enabled=use_amp)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log("\n[MODEL]")
    log(f"Total params         : {total_params:,}")
    log(f"Trainable params     : {trainable_params:,}")
    log(f"Backbone lr          : {BACKBONE_LR}")
    log(f"Head lr              : {HEAD_LR}")
    log(f"Weight decay         : {WEIGHT_DECAY}")
    log(f"Dropout              : {DROPOUT}")
    log(f"Random erasing       : {RANDOM_ERASE_PROB}")
    log(f"Light MixUp          : {USE_LIGHT_MIXUP}")
    log(f"MixUp prob           : {MIXUP_PROB}")
    log(f"EMA                  : {USE_EMA}")
    log(f"Label smoothing      : {LABEL_SMOOTHING}")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "val_bal_acc": []}
    best_val_loss = float("inf")
    patience_count = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_dir = CHECKPOINT_DIR / ARTIFACTS_DIRNAME
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME

    log("\n[TRAINING START]")
    for epoch in range(1, EPOCHS + 1):
        scheduler.step(epoch - 1)
        optimizer = maybe_unfreeze_all(model, optimizer, epoch, unfreeze_epoch=UNFREEZE_EPOCH)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        log("-" * 80)
        log(f"Epoch {epoch:02d}/{EPOCHS} | lrs={[f'{lr:.7f}' for lr in current_lrs]}")

        tr_loss, tr_acc, tr_bal_acc, _, _, _, _ = run_epoch(
            train_loader, model, criterion, optimizer, scaler, device,
            train=True, epoch_num=epoch, stage_name="TRAIN", threshold=THRESHOLD_DEFAULT
        )

        va_loss, va_acc, _, y_val_true, y_val_prob, _, _ = run_epoch(
            val_loader, model, criterion, optimizer, scaler, device,
            train=False, epoch_num=epoch, stage_name="VAL", threshold=THRESHOLD_DEFAULT
        )

        best_threshold, best_val_bal, _ = find_best_threshold(y_val_true, y_val_prob)

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(va_acc)
        history["val_bal_acc"].append(best_val_bal)

        log(
            f"[EPOCH SUMMARY] {epoch:02d}/{EPOCHS} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_bal_acc={tr_bal_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc@0.5={va_acc:.4f} val_bal_acc@best_thr={best_val_bal:.4f} "
            f"best_thr={best_threshold:.2f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_count = 0
            save_obj = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_threshold": best_threshold,
                "config": {
                    "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "backbone_lr": BACKBONE_LR,
                    "head_lr": HEAD_LR,
                    "dropout": DROPOUT,
                    "seed": SEED,
                    "root_dir": str(root_dir),
                    "num_workers": NUM_WORKERS,
                    "mixup_prob": MIXUP_PROB,
                },
            }
            torch.save(save_obj, checkpoint_path)
            log(f"[CHECKPOINT] Saved new best model at epoch {epoch:02d} -> {checkpoint_path}")
        else:
            patience_count += 1
            log(f"[EARLY STOP WATCH] No improvement count = {patience_count}/{EARLY_STOP_PAT}")
            if patience_count >= EARLY_STOP_PAT:
                log(f"[EARLY STOP] Stopping at epoch {epoch:02d}")
                break

    if not checkpoint_path.exists():
        raise RuntimeError("No checkpoint was saved.")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_threshold = float(ckpt.get("best_threshold", THRESHOLD_DEFAULT))

    log("\n[BEST MODEL RESTORED]")
    log(f"Best epoch           : {ckpt['epoch']}")
    log(f"Best val loss        : {ckpt['best_val_loss']:.4f}")
    log(f"Best val threshold   : {best_threshold:.2f}")

    log("\n[TEST EVALUATION]")
    test_loss, test_acc, _, y_true, y_prob, _, test_paths_out = run_epoch(
        test_loader, model, criterion, optimizer, scaler, device,
        train=False, epoch_num=None, stage_name="TEST", threshold=THRESHOLD_DEFAULT
    )
    y_pred = (y_prob >= best_threshold).astype(int)
    test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    report_text = classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)

    alt_thresholds = [0.30, 0.40, 0.50, 0.60, float(best_threshold)]
    threshold_table = evaluate_thresholds(y_true, y_prob, alt_thresholds)

    log(f"Test loss            : {test_loss:.4f}")
    log(f"Test acc@0.5         : {test_acc:.4f}")
    log(f"Test acc@best_thr    : {(y_pred == y_true).mean():.4f}")
    log(f"Test balanced acc    : {test_bal_acc:.4f}")
    log(f"ROC AUC              : {roc_auc:.4f}")
    log(f"Average precision    : {ap:.4f}")
    log(f"Applied threshold    : {best_threshold:.2f}")
    log("\n[THRESHOLD CHECKS]")
    for row in threshold_table:
        log(
            f"thr={row['threshold']:.2f} | "
            f"acc={row['accuracy']:.4f} | "
            f"bal_acc={row['balanced_accuracy']:.4f} | "
            f"cm={row['cm']}"
        )

    log("\n[CLASSIFICATION REPORT]")
    log(report_text)
    log("[CONFUSION MATRIX]")
    log(str(confusion_matrix(y_true, y_pred)))

    save_artifacts(artifacts_dir, history, y_true, y_prob, y_pred, best_threshold, report_text)

    with open(artifacts_dir / "threshold_checks.txt", "w", encoding="utf-8") as f:
        for row in threshold_table:
            f.write(
                f"thr={row['threshold']:.2f},acc={row['accuracy']:.6f},"
                f"bal_acc={row['balanced_accuracy']:.6f},cm={row['cm']}\n"
            )

    with open(artifacts_dir / "split_debug.txt", "w", encoding="utf-8") as f:
        f.write(f"root_dir={root_dir}\n")
        f.write(f"train_real_dir={train_real_dir}\n")
        f.write(f"train_fake_dir={train_fake_dir}\n")
        f.write(f"test_real_dir={test_real_dir}\n")
        f.write(f"test_fake_dir={test_fake_dir}\n")
        f.write(f"train_samples={len(tr_paths)} real={tr_real} fake={tr_fake}\n")
        f.write(f"val_samples={len(va_paths)} real={va_real} fake={va_fake}\n")
        f.write(f"test_samples={len(test_paths)} real={te_real} fake={te_fake}\n")
        f.write(f"best_threshold={best_threshold:.4f}\n")

    with open(artifacts_dir / "test_predictions.csv", "w", encoding="utf-8") as f:
        f.write("path,true_label,prob_fake,pred_best_thr\n")
        for p, yt, yp, yd in zip(test_paths_out, y_true, y_prob, y_pred):
            f.write(f"{p},{int(yt)},{float(yp):.8f},{int(yd)}\n")

    log("\n[SAVED ARTIFACTS]")
    log(f"Metrics summary      : {(artifacts_dir / 'metrics_summary.txt').resolve()}")
    log(f"Threshold checks     : {(artifacts_dir / 'threshold_checks.txt').resolve()}")
    log(f"Loss curve           : {(artifacts_dir / 'loss_curve.png').resolve()}")
    log(f"Accuracy curve       : {(artifacts_dir / 'accuracy_curve.png').resolve()}")
    log(f"Confusion matrix     : {(artifacts_dir / 'confusion_matrix.png').resolve()}")
    log(f"ROC curve            : {(artifacts_dir / 'roc_curve.png').resolve()}")
    log(f"PR curve             : {(artifacts_dir / 'pr_curve.png').resolve()}")
    log(f"Score distribution   : {(artifacts_dir / 'score_distribution.png').resolve()}")
    log(f"Test predictions CSV : {(artifacts_dir / 'test_predictions.csv').resolve()}")

    log("\n[RUN COMPLETE]")
    log(f"Checkpoint saved at  : {checkpoint_path.resolve()}")
    log(f"Artifacts saved at   : {artifacts_dir.resolve()}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
