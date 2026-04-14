import os
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
from torchvision.models import swin_t, Swin_T_Weights

warnings.filterwarnings("ignore")

SEED = 42
BATCH_SIZE = 24
EPOCHS = 10
EARLY_STOP_PAT = 2
IMG_SIZE = 224
NUM_WORKERS = 0
VAL_SIZE = 0.2
DROPOUT = 0.35
LR = 5e-5
HEAD_LR_MULTIPLIER = 5.0
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6
UNFREEZE_LAST_STAGES = 2
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.0
THRESHOLD_GRID = np.round(np.arange(0.20, 0.801, 0.02), 2)
CHECKPOINT_DIR = Path("models/swin/stable_run")
CHECKPOINT_NAME = "best_swin_t_stable_v2.pt"
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
        return img, torch.tensor(label, dtype=torch.float32)



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
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    log(f"[LOADER] {loader_name:<10} batches={len(loader)} samples={len(ds)} shuffle={shuffle}")
    return loader



def build_model(weights):
    model = swin_t(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    stages = list(model.features)
    start_idx = max(0, len(stages) - UNFREEZE_LAST_STAGES)
    for stage in stages[start_idx:]:
        for param in stage.parameters():
            param.requires_grad = True

    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 1),
    )

    for param in model.head.parameters():
        param.requires_grad = True

    return model



def build_optimizer(model):
    head_params = []
    body_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head"):
            head_params.append(param)
        else:
            body_params.append(param)

    param_groups = []
    if body_params:
        param_groups.append({"params": body_params, "lr": LR})
    if head_params:
        param_groups.append({"params": head_params, "lr": LR * HEAD_LR_MULTIPLIER})

    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)



def run_epoch(model, loader, criterion, optimizer, scaler, threshold, train=True, epoch_num=None):
    mode = "TRAIN" if train else "EVAL"
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_probs = []
    all_true = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=f"Epoch {epoch_num:02d} {mode}" if epoch_num is not None else mode, leave=False)
        for batch_idx, (imgs, labels) in enumerate(pbar, start=1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=USE_AMP):
                logits = model(imgs).squeeze(1)
                if LABEL_SMOOTHING > 0 and train:
                    smooth_labels = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
                    loss = criterion(logits, smooth_labels)
                else:
                    loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            total_correct += (preds == labels.long()).sum().item()
            total_seen += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            all_probs.extend(probs.detach().cpu().numpy())
            all_true.extend(labels.detach().cpu().numpy())

            running_loss = total_loss / max(total_seen, 1)
            running_acc = total_correct / max(total_seen, 1)
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                log(
                    f"[{mode}] epoch={epoch_num:02d} batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={running_loss:.4f} acc={running_acc:.4f}"
                )

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    y_true = np.array(all_true).astype(int)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return avg_loss, avg_acc, bal_acc, y_true, y_prob, y_pred



def save_curves(output_dir, history, cm, fpr, tpr, roc_auc, precision, recall, ap, y_true, y_prob, threshold):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
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



def main():
    ROOT_DIR = detect_root()
    TRAIN_REAL_DIR = find_class_dir(ROOT_DIR, "train", "real")
    TRAIN_FAKE_DIR = find_class_dir(ROOT_DIR, "train", "fake")
    TEST_REAL_DIR = find_class_dir(ROOT_DIR, "test", "real")
    TEST_FAKE_DIR = find_class_dir(ROOT_DIR, "test", "fake")

    log("=" * 80)
    log("SWIN-T STABLE TRAINING RUN")
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
    log(f"Base learning rate   : {LR}")
    log(f"Head LR multiplier   : {HEAD_LR_MULTIPLIER}")
    log(f"Weight decay         : {WEIGHT_DECAY}")
    log(f"Dropout              : {DROPOUT}")
    log(f"Unfreeze last stages : {UNFREEZE_LAST_STAGES}")
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
    log("\n[DEBUG SAMPLE PATHS]")
    for i, p in enumerate(TR_PATHS[:3]):
        log(f"Train sample {i + 1}      : {p}")
    for i, p in enumerate(VA_PATHS[:3]):
        log(f"Val sample {i + 1}        : {p}")
    for i, p in enumerate(test_paths[:3]):
        log(f"Test sample {i + 1}       : {p}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    split_file = CHECKPOINT_DIR / "split_debug.txt"
    with open(split_file, "w", encoding="utf-8") as f:
        f.write("TRAIN\n")
        for p, y in zip(TR_PATHS[:300], TR_LABELS[:300]):
            f.write(f"{y}\t{p}\n")
        f.write("\nVAL\n")
        for p, y in zip(VA_PATHS[:300], VA_LABELS[:300]):
            f.write(f"{y}\t{p}\n")
        f.write("\nTEST\n")
        for p, y in zip(test_paths[:300], test_labels[:300]):
            f.write(f"{y}\t{p}\n")

    weights = Swin_T_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_loader = make_loader(TR_PATHS, TR_LABELS, train_transform, True, "train")
    val_loader = make_loader(VA_PATHS, VA_LABELS, eval_transform, False, "val")
    test_loader = make_loader(test_paths, test_labels, eval_transform, False, "test")

    model = build_model(weights).to(DEVICE)

    num_pos = sum(TR_LABELS)
    num_neg = len(TR_LABELS) - num_pos
    pos_weight_value = num_neg / max(num_pos, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=MIN_LR,
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log("\n[MODEL]")
    log(f"Total params         : {total_params:,}")
    log(f"Trainable params     : {trainable_params:,}")
    log(f"Positive class wt    : {pos_weight_value:.6f}")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "val_bal_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1
    best_threshold = 0.5
    patience_count = 0
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME

    log("\n[TRAINING START]")
    for epoch in range(1, EPOCHS + 1):
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        log("-" * 80)
        log(f"Epoch {epoch:02d}/{EPOCHS} | lrs={[f'{x:.7f}' for x in current_lrs]}")

        tr_loss, tr_acc, tr_bal_acc, _, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            threshold=0.5,
            train=True,
            epoch_num=epoch,
        )
        va_loss, _, _, va_true, va_prob, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            threshold=0.5,
            train=False,
            epoch_num=epoch,
        )

        val_threshold, val_bal_acc, val_acc = search_best_threshold(va_true, va_prob)
        scheduler.step(va_loss)

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_threshold": best_threshold,
                    "config": {
                        "img_size": IMG_SIZE,
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "base_lr": LR,
                        "head_lr_multiplier": HEAD_LR_MULTIPLIER,
                        "dropout": DROPOUT,
                        "seed": SEED,
                        "root_dir": str(ROOT_DIR),
                        "unfreeze_last_stages": UNFREEZE_LAST_STAGES,
                    },
                },
                checkpoint_path,
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
    model.load_state_dict(ckpt["model_state_dict"])
    best_threshold = float(ckpt.get("best_threshold", 0.5))
    best_epoch = int(ckpt["epoch"])

    log("\n[BEST MODEL RESTORED]")
    log(f"Best epoch           : {best_epoch}")
    log(f"Best val loss        : {ckpt['best_val_loss']:.4f}")
    log(f"Best val threshold   : {best_threshold:.2f}")

    log("\n[TEST EVALUATION]")
    test_loss, _, _, y_true, y_prob, _ = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        threshold=best_threshold,
        train=False,
        epoch_num=best_epoch,
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

    save_curves(
        output_dir=CHECKPOINT_DIR,
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

    with open(CHECKPOINT_DIR / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss={ckpt['best_val_loss']:.6f}\n")
        f.write(f"best_threshold={best_threshold:.2f}\n")
        f.write(f"test_loss={test_loss:.6f}\n")
        f.write(f"test_acc={test_acc:.6f}\n")
        f.write(f"test_bal_acc={test_bal_acc:.6f}\n")
        f.write(f"roc_auc={roc_auc:.6f}\n")
        f.write(f"average_precision={ap:.6f}\n")
        f.write(f"confusion_matrix={cm.tolist()}\n")

    log("\n[RUN COMPLETE]")
    log(f"Checkpoint saved at  : {checkpoint_path.resolve()}")
    log(f"Artifacts saved at   : {CHECKPOINT_DIR.resolve()}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
