
import csv
import os
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

warnings.filterwarnings("ignore")

SEED = 42
BATCH_SIZE = 128
EPOCHS = 8
EARLY_STOP_PAT = 2
IMG_SIZE = 224
NUM_WORKERS = 4
VAL_SIZE = 0.2
DROPOUT = 0.2
BACKBONE_LR = 1e-4
HEAD_LR = 5e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
UNFREEZE_EPOCH = 4

CHECKPOINT_DIR = Path("models/efficientnet/base_improved_run")
CHECKPOINT_NAME = "best_efficientnet_b0_base_improved.pt"
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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


def log(msg=""):
    print(msg, flush=True)


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
        except Exception as e:
            log(f"[WARN] Failed to read image: {path} | {e}")
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return img, torch.tensor(label, dtype=torch.float32), str(path)


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
        "Could not find dataset root. Set DEEPNET_ROOT or place your data in one of these structures:\n"
        "data/deepdetect-2025_dddata/train/{real,fake} and test/{real,fake}\n"
        "or data/deepdetect-2025/train/{real,fake} and test/{real,fake}"
    )


def count_labels(labels):
    labels = np.array(labels)
    real_count = int((labels == 0).sum())
    fake_count = int((labels == 1).sum())
    return real_count, fake_count


def make_loader(paths, labels, transform, shuffle, loader_name):
    ds = DeepfakeDataset(paths, labels, transform)
    loader_kwargs = dict(
        dataset=ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)
    log(f"[LOADER] {loader_name:<10} batches={len(loader)} samples={len(ds)} shuffle={shuffle}")
    return loader


def build_model(dropout=0.2):
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


def build_optimizer(model):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )


def maybe_unfreeze_all(model, optimizer, scheduler, current_epoch):
    if current_epoch != UNFREEZE_EPOCH:
        return optimizer, scheduler

    log("[FINE-TUNE] Unfreezing full backbone from this epoch onward")
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    log("[FINE-TUNE] Rebuilt optimizer and scheduler for full-backbone training")
    return optimizer, scheduler


def run_epoch(loader, model, criterion, optimizer, scaler, train=True, epoch_num=None, stage_name=None, threshold=0.5):
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

            running_loss = total_loss / total_seen
            running_acc = total_correct / total_seen
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                epoch_part = f"epoch={epoch_num:02d} " if epoch_num is not None else ""
                log(
                    f"[{stage_name}] {epoch_part}batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={running_loss:.4f} acc={running_acc:.4f}"
                )

    avg_loss = total_loss / total_seen
    avg_acc = total_correct / total_seen
    y_true = np.array(all_true).astype(int)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return avg_loss, avg_acc, bal_acc, y_true, y_prob, y_pred, all_paths


def threshold_diagnostics(y_true, y_prob, thresholds=(0.30, 0.40, 0.50, 0.60)):
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        rows.append(
            {
                "threshold": thr,
                "acc": float((y_pred == y_true).mean()),
                "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
                "cm": confusion_matrix(y_true, y_pred).tolist(),
            }
        )
    return rows


def save_artifacts(output_dir, history, y_true, y_prob, y_pred, threshold, report_text, threshold_rows):
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

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
        f.write("\n\nthreshold_checks\n")
        for row in threshold_rows:
            f.write(f"thr={row['threshold']:.2f}, acc={row['acc']:.6f}, bal_acc={row['bal_acc']:.6f}, cm={row['cm']}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["accuracy"], label="Train")
    axes[1].plot(history["val_accuracy"], label="Val", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
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
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.legend()
    plt.grid(True)
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
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=200)
    plt.close()


def main():
    ROOT_DIR = detect_root()
    TRAIN_REAL_DIR = find_class_dir(ROOT_DIR, "train", "real")
    TRAIN_FAKE_DIR = find_class_dir(ROOT_DIR, "train", "fake")
    TEST_REAL_DIR = find_class_dir(ROOT_DIR, "test", "real")
    TEST_FAKE_DIR = find_class_dir(ROOT_DIR, "test", "fake")

    log("=" * 80)
    log("EFFICIENTNET B0 BASE IMPROVED RUN")
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
    log(f"Checkpoint           : {CHECKPOINT_DIR / CHECKPOINT_NAME}")
    if torch.cuda.device_count() > 1:
        log(f"GPUs available       : {torch.cuda.device_count()}")

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
    log("\n[DEBUG SAMPLE PATHS]")
    for i, p in enumerate(TR_PATHS[:3]):
        log(f"Train sample {i + 1}      : {p}")
    for i, p in enumerate(VA_PATHS[:3]):
        log(f"Val sample {i + 1}        : {p}")
    for i, p in enumerate(test_paths[:3]):
        log(f"Test sample {i + 1}       : {p}")

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_loader = make_loader(TR_PATHS, TR_LABELS, train_transform, True, "train")
    val_loader = make_loader(VA_PATHS, VA_LABELS, eval_transform, False, "val")
    test_loader = make_loader(test_paths, test_labels, eval_transform, False, "test")

    model = build_model(dropout=DROPOUT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    scaler = GradScaler("cuda", enabled=USE_AMP)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log("\n[MODEL]")
    log(f"Total params         : {total_params:,}")
    log(f"Trainable params     : {trainable_params:,}")
    log(f"Backbone lr          : {BACKBONE_LR}")
    log(f"Head lr              : {HEAD_LR}")
    log(f"Weight decay         : {WEIGHT_DECAY}")
    log(f"Dropout              : {DROPOUT}")
    log(f"Threshold            : {THRESHOLD}")
    log(f"Unfreeze epoch       : {UNFREEZE_EPOCH}")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    patience_count = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_dir = CHECKPOINT_DIR / ARTIFACTS_DIRNAME
    CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME

    log("\n[TRAINING START]")
    for epoch in range(1, EPOCHS + 1):
        optimizer, scheduler = maybe_unfreeze_all(model, optimizer, scheduler, epoch)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        log("-" * 80)
        log(f"Epoch {epoch:02d}/{EPOCHS} | lrs={[f'{lr:.7f}' for lr in current_lrs]}")

        tr_loss, tr_acc, tr_bal_acc, _, _, _, _ = run_epoch(
            train_loader, model, criterion, optimizer, scaler,
            train=True, epoch_num=epoch, stage_name="TRAIN", threshold=THRESHOLD
        )
        va_loss, va_acc, va_bal_acc, _, _, _, _ = run_epoch(
            val_loader, model, criterion, optimizer, scaler,
            train=False, epoch_num=epoch, stage_name="VAL", threshold=THRESHOLD
        )

        scheduler.step(va_loss)

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(va_acc)

        log(
            f"[EPOCH SUMMARY] {epoch:02d}/{EPOCHS} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_bal_acc={tr_bal_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_bal_acc={va_bal_acc:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": {
                        "img_size": IMG_SIZE,
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "backbone_lr": BACKBONE_LR,
                        "head_lr": HEAD_LR,
                        "dropout": DROPOUT,
                        "threshold": THRESHOLD,
                        "seed": SEED,
                        "root_dir": str(ROOT_DIR),
                        "num_workers": NUM_WORKERS,
                        "unfreeze_epoch": UNFREEZE_EPOCH,
                    },
                },
                CHECKPOINT_PATH,
            )
            log(f"[CHECKPOINT] Saved new best model at epoch {epoch:02d} -> {CHECKPOINT_PATH}")
        else:
            patience_count += 1
            log(f"[EARLY STOP WATCH] No improvement count = {patience_count}/{EARLY_STOP_PAT}")
            if patience_count >= EARLY_STOP_PAT:
                log(f"[EARLY STOP] Stopping at epoch {epoch:02d}")
                break

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError("No checkpoint was saved. Training may have failed early.")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    log("\n[BEST MODEL RESTORED]")
    log(f"Best epoch           : {ckpt['epoch']}")
    log(f"Best val loss        : {ckpt['best_val_loss']:.4f}")

    log("\n[TEST EVALUATION]")
    test_loss, test_acc, test_bal_acc, y_true, y_prob, y_pred, test_paths_out = run_epoch(
        test_loader, model, criterion, optimizer, scaler,
        train=False, epoch_num=None, stage_name="TEST", threshold=THRESHOLD
    )

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    report_text = classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)
    thr_rows = threshold_diagnostics(y_true, y_prob)

    log(f"Test loss            : {test_loss:.4f}")
    log(f"Test acc             : {test_acc:.4f}")
    log(f"Test balanced acc    : {test_bal_acc:.4f}")
    log(f"ROC AUC              : {roc_auc:.4f}")
    log(f"Average precision    : {ap:.4f}")
    log("\n[THRESHOLD CHECKS]")
    for row in thr_rows:
        log(
            f"thr={row['threshold']:.2f} | "
            f"acc={row['acc']:.4f} | "
            f"bal_acc={row['bal_acc']:.4f} | "
            f"cm={row['cm']}"
        )
    log("\n[CLASSIFICATION REPORT]")
    log(report_text)
    log("[CONFUSION MATRIX]")
    log(str(cm))

    save_artifacts(artifacts_dir, history, y_true, y_prob, y_pred, THRESHOLD, report_text, thr_rows)

    with open(artifacts_dir / "training_history.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for i, (trl, tra, val_l, val_a) in enumerate(zip(history["loss"], history["accuracy"], history["val_loss"], history["val_accuracy"]), start=1):
            writer.writerow([i, f"{trl:.8f}", f"{tra:.8f}", f"{val_l:.8f}", f"{val_a:.8f}"])

    with open(artifacts_dir / "threshold_diagnostics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "accuracy", "balanced_accuracy", "confusion_matrix"])
        for row in thr_rows:
            writer.writerow([f"{row['threshold']:.2f}", f"{row['acc']:.8f}", f"{row['bal_acc']:.8f}", str(row["cm"])])

    with open(artifacts_dir / "run_config.txt", "w", encoding="utf-8") as f:
        f.write(f"seed={SEED}\n")
        f.write(f"batch_size={BATCH_SIZE}\n")
        f.write(f"epochs={EPOCHS}\n")
        f.write(f"early_stop_pat={EARLY_STOP_PAT}\n")
        f.write(f"img_size={IMG_SIZE}\n")
        f.write(f"num_workers={NUM_WORKERS}\n")
        f.write(f"val_size={VAL_SIZE}\n")
        f.write(f"dropout={DROPOUT}\n")
        f.write(f"backbone_lr={BACKBONE_LR}\n")
        f.write(f"head_lr={HEAD_LR}\n")
        f.write(f"weight_decay={WEIGHT_DECAY}\n")
        f.write(f"threshold={THRESHOLD}\n")
        f.write(f"unfreeze_epoch={UNFREEZE_EPOCH}\n")
        f.write(f"checkpoint_dir={CHECKPOINT_DIR}\n")
        f.write(f"checkpoint_name={CHECKPOINT_NAME}\n")
        f.write(f"dataset_root={ROOT_DIR}\n")

    with open(artifacts_dir / "test_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "prob_fake", "pred_0.5"])
        for p, yt, yp, yd in zip(test_paths_out, y_true, y_prob, y_pred):
            writer.writerow([p, int(yt), f"{float(yp):.8f}", int(yd)])

    log("\n[SAVED ARTIFACTS]")
    log(f"Metrics summary      : {(artifacts_dir / 'metrics_summary.txt').resolve()}")
    log(f"Training curves      : {(artifacts_dir / 'training_curves.png').resolve()}")
    log(f"Confusion matrix     : {(artifacts_dir / 'confusion_matrix.png').resolve()}")
    log(f"ROC curve            : {(artifacts_dir / 'roc_curve.png').resolve()}")
    log(f"PR curve             : {(artifacts_dir / 'pr_curve.png').resolve()}")
    log(f"Score distribution   : {(artifacts_dir / 'score_distribution.png').resolve()}")
    log(f"Training history     : {(artifacts_dir / 'training_history.csv').resolve()}")
    log(f"Threshold CSV        : {(artifacts_dir / 'threshold_diagnostics.csv').resolve()}")
    log(f"Run config           : {(artifacts_dir / 'run_config.txt').resolve()}")
    log(f"Test predictions CSV : {(artifacts_dir / 'test_predictions.csv').resolve()}")

    log("\n[RUN COMPLETE]")
    log(f"Checkpoint saved at  : {CHECKPOINT_PATH.resolve()}")
    log(f"Artifacts saved at   : {artifacts_dir.resolve()}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
