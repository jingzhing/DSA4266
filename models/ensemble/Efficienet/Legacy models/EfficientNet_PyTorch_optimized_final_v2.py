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
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

warnings.filterwarnings("ignore")

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 12
EARLY_STOP_PAT = 2
NUM_WORKERS = 0  # safer on Windows and prevents repeated startup prints
VAL_SIZE = 0.2

BACKBONE_LR = 1e-4
HEAD_LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05

RANDOM_ERASE_PROB = 0.15
USE_EMA = True
EMA_DECAY = 0.999
THRESHOLD_DEFAULT = 0.5
THRESHOLD_SEARCH_MIN = 0.20
THRESHOLD_SEARCH_MAX = 0.80
THRESHOLD_SEARCH_STEP = 0.01

CHECKPOINT_DIR = Path("models/efficientnet/optimized_run")
CHECKPOINT_NAME = "best_efficientnet_b0_optimized_final.pt"
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
        return img, torch.tensor(label, dtype=torch.float32)


class MixupCutmixCollator:
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=0.5, prob=0.6):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def _rand_bbox(self, size, lam):
        _, _, h, w = size
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        return x1, y1, x2, y2

    def __call__(self, batch):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.stack([b[1] for b in batch], dim=0)

        if len(batch) < 2 or np.random.rand() > self.prob:
            return imgs, labels, labels, 1.0

        indices = torch.randperm(imgs.size(0))
        shuffled_imgs = imgs[indices]
        shuffled_labels = labels[indices]

        if np.random.rand() < 0.5:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            imgs = lam * imgs + (1.0 - lam) * shuffled_imgs
            return imgs, labels, shuffled_labels, float(lam)

        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        x1, y1, x2, y2 = self._rand_bbox(imgs.size(), lam)
        imgs[:, :, y1:y2, x1:x2] = shuffled_imgs[:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (imgs.size(-1) * imgs.size(-2)))
        return imgs, labels, shuffled_labels, float(lam)


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


class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = float(smoothing)
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.loss(logits, targets).mean()


class ModelEma:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.module = AveragedModel(model, avg_fn=lambda avg, cur, n: decay * avg + (1 - decay) * cur)

    def update(self, model):
        self.module.update_parameters(model)



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
        transforms.Resize((236, 236)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
        transforms.RandomErasing(p=RANDOM_ERASE_PROB, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    return train_transform, eval_transform



def make_loader(paths, labels, transform, shuffle, loader_name, collate_fn=None):
    ds = DeepfakeDataset(paths, labels, transform)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    log(f"[LOADER] {loader_name:<10} batches={len(loader)} samples={len(ds)} shuffle={shuffle}")
    return loader



def build_model(dropout=0.3):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    # unfreeze later feature blocks only
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

    # keep momentum states if shapes match
    try:
        new_optimizer.load_state_dict(optimizer.state_dict())
        # restore intended lrs
        new_optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]
        new_optimizer.param_groups[1]["lr"] = optimizer.param_groups[1]["lr"]
    except Exception:
        pass
    return new_optimizer



def run_epoch(loader, model, criterion, optimizer, scaler, device, train=True, epoch_num=None, stage_name=None, threshold=0.5, ema=None):
    stage_name = stage_name or ("TRAIN" if train else "EVAL")
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_probs = []
    all_true = []

    desc = f"Epoch {epoch_num:02d} {stage_name}" if epoch_num is not None else stage_name
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(pbar, start=1):
            if train:
                imgs, labels_a, labels_b, lam = batch
                labels_a = labels_a.to(device, non_blocking=True)
                labels_b = labels_b.to(device, non_blocking=True)
                labels = labels_a
            else:
                imgs, labels = batch
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
                if ema is not None:
                    ema.update(model)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            batch_correct = (preds == labels.long()).sum().item()

            total_correct += batch_correct
            total_seen += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            all_probs.extend(probs.detach().cpu().numpy())
            all_true.extend(labels.detach().cpu().numpy())

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
    return avg_loss, avg_acc, bal_acc, y_true, y_prob, y_pred



def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(THRESHOLD_SEARCH_MIN, THRESHOLD_SEARCH_MAX + 1e-9, THRESHOLD_SEARCH_STEP)
    best_threshold = THRESHOLD_DEFAULT
    best_score = -1.0
    records = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        records.append((thr, bal_acc))
        if bal_acc > best_score:
            best_score = bal_acc
            best_threshold = float(thr)

    return best_threshold, best_score, records



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
    log("EFFICIENTNET B0 OPTIMIZED TRAINING RUN")
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
    train_loader = make_loader(tr_paths, tr_labels, train_transform, True, "train", collate_fn=MixupCutmixCollator())
    val_loader = make_loader(va_paths, va_labels, eval_transform, False, "val")
    test_loader = make_loader(test_paths, test_labels, eval_transform, False, "test")
    bn_loader = make_loader(tr_paths, tr_labels, eval_transform, False, "bn_update") if USE_EMA else None

    model = build_model(dropout=DROPOUT).to(device)
    criterion = SmoothedBCEWithLogitsLoss(smoothing=LABEL_SMOOTHING)
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
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=1, total_epochs=EPOCHS, min_lr_ratio=0.05)
    scaler = GradScaler("cuda", enabled=use_amp)
    ema = ModelEma(model, decay=EMA_DECAY) if USE_EMA else None

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log("\n[MODEL]")
    log(f"Total params         : {total_params:,}")
    log(f"Trainable params     : {trainable_params:,}")
    log(f"Backbone lr          : {BACKBONE_LR}")
    log(f"Head lr              : {HEAD_LR}")
    log(f"Weight decay         : {WEIGHT_DECAY}")
    log(f"Dropout              : {DROPOUT}")
    log(f"Label smoothing      : {LABEL_SMOOTHING}")
    log(f"Random erasing       : {RANDOM_ERASE_PROB}")
    log(f"EMA                  : {USE_EMA}")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "val_bal_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1
    patience_count = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_dir = CHECKPOINT_DIR / ARTIFACTS_DIRNAME
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME

    log("\n[TRAINING START]")
    for epoch in range(1, EPOCHS + 1):
        scheduler.step(epoch - 1)
        optimizer = maybe_unfreeze_all(model, optimizer, epoch, unfreeze_epoch=4)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        log("-" * 80)
        log(f"Epoch {epoch:02d}/{EPOCHS} | lrs={[f'{lr:.7f}' for lr in current_lrs]}")

        tr_loss, tr_acc, tr_bal_acc, _, _, _ = run_epoch(
            train_loader, model, criterion, optimizer, scaler, device,
            train=True, epoch_num=epoch, stage_name="TRAIN", threshold=THRESHOLD_DEFAULT, ema=ema
        )

        eval_model = ema.module.module if ema is not None else model
        va_loss, va_acc, va_bal_acc, y_val_true, y_val_prob, _ = run_epoch(
            val_loader, eval_model, criterion, optimizer, scaler, device,
            train=False, epoch_num=epoch, stage_name="VAL", threshold=THRESHOLD_DEFAULT, ema=None
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

        improved = va_loss < best_val_loss
        if improved:
            best_val_loss = va_loss
            best_epoch = epoch
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
                    "early_stop_pat": EARLY_STOP_PAT,
                },
            }
            if ema is not None:
                save_obj["ema_state_dict"] = ema.module.module.state_dict()
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
    if ema is not None and "ema_state_dict" in ckpt:
        ema.module.module.load_state_dict(ckpt["ema_state_dict"])
        eval_model = ema.module.module
        if bn_loader is not None:
            log("[EMA] Updating BatchNorm stats on EMA model")
            eval_model.train()
            update_bn(bn_loader, eval_model, device=device)
            eval_model.eval()
    else:
        eval_model = model

    best_threshold = float(ckpt.get("best_threshold", THRESHOLD_DEFAULT))
    log("\n[BEST MODEL RESTORED]")
    log(f"Best epoch           : {ckpt['epoch']}")
    log(f"Best val loss        : {ckpt['best_val_loss']:.4f}")
    log(f"Best val threshold   : {best_threshold:.2f}")

    log("\n[TEST EVALUATION]")
    test_loss, test_acc, _, y_true, y_prob, _ = run_epoch(
        test_loader, eval_model, criterion, optimizer, scaler, device,
        train=False, epoch_num=None, stage_name="TEST", threshold=THRESHOLD_DEFAULT, ema=None
    )
    y_pred = (y_prob >= best_threshold).astype(int)
    test_bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    report_text = classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)

    log(f"Test loss            : {test_loss:.4f}")
    log(f"Test acc@0.5         : {test_acc:.4f}")
    log(f"Test acc@best_thr    : {(y_pred == y_true).mean():.4f}")
    log(f"Test balanced acc    : {test_bal_acc:.4f}")
    log(f"ROC AUC              : {roc_auc:.4f}")
    log(f"Average precision    : {ap:.4f}")
    log(f"Applied threshold    : {best_threshold:.2f}")
    log("\n[CLASSIFICATION REPORT]")
    log(report_text)
    log("[CONFUSION MATRIX]")
    log(str(confusion_matrix(y_true, y_pred)))

    save_artifacts(artifacts_dir, history, y_true, y_prob, y_pred, best_threshold, report_text)

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

    log("\n[RUN COMPLETE]")
    log(f"Checkpoint saved at  : {checkpoint_path.resolve()}")
    log(f"Artifacts saved at   : {artifacts_dir.resolve()}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
