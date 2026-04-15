import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 128
EPOCHS = 8
EARLY_STOP_PAT = 5
IMG_SIZE = 224
NUM_WORKERS = 4
VAL_SIZE = 0.2

VALID_EXT = (".jpg", ".jpeg", ".png")
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


# ── Dataset Detection ──────────────────────────────────────────────────────────
def get_image_paths(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return [str(p) for p in directory.glob("*") if p.suffix.lower() in VALID_EXT]


def find_class_dir(root_dir, split_name, class_name):
    root_dir = Path(root_dir)
    direct = root_dir / split_name / class_name
    if direct.exists() and direct.is_dir():
        return direct

    split_lower = split_name.lower()
    class_lower = class_name.lower()

    for p in root_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == class_lower and p.parent.name.lower() == split_lower:
            return p

    return direct


def detect_root():
    env_root = None
    try:
        import os
        env_root = os.environ.get("DEEPNET_ROOT")
    except Exception:
        env_root = None

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
        "Could not find dataset root. Set DEEPNET_ROOT or place your data under a standard structure like "
        "data/deepdetect-2025_dddata/train/{real,fake} and test/{real,fake}."
    )


# ── Data Augmentation Pipeline ─────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


class DeepfakeDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


def make_loader(paths, labels, transform, shuffle):
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
    return DataLoader(**loader_kwargs)


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(dropout=0.3):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    all_layers = list(model.named_modules())
    non_bn_layers = [
        name for name, m in all_layers
        if not isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
        and len(list(m.children())) == 0
        and len(list(m.parameters())) > 0
    ]
    for name in non_bn_layers[-30:]:
        for n, m in model.named_modules():
            if n == name:
                for param in m.parameters():
                    param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )

    return model


# ── Training / Eval ────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, train=True, epoch_num=None, stage_name=None):
    stage_name = stage_name or ("TRAIN" if train else "VAL")
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = f"Epoch {epoch_num:02d} {stage_name}" if epoch_num is not None else stage_name

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch_idx, (imgs, labels) in enumerate(pbar, start=1):
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

            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_seen += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            running_loss = total_loss / total_seen
            running_acc = total_correct / total_seen
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                epoch_part = f"epoch={epoch_num:02d} " if epoch_num is not None else ""
                print(
                    f"[{stage_name}] {epoch_part}batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={running_loss:.4f} acc={running_acc:.4f}",
                    flush=True,
                )

    return total_loss / total_seen, total_correct / total_seen


def evaluate_test(model, loader, criterion):
    model.eval()
    all_probs, all_true = [], []
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="TEST", leave=False, dynamic_ncols=True)
        for batch_idx, (imgs, labels) in enumerate(pbar, start=1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast("cuda", enabled=USE_AMP):
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            total_correct += (preds == labels.long()).sum().item()
            total_seen += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

            running_loss = total_loss / total_seen
            running_acc = total_correct / total_seen
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                print(
                    f"[TEST] batch={batch_idx:04d}/{len(loader):04d} loss={running_loss:.4f} acc={running_acc:.4f}",
                    flush=True,
                )

    y_prob = np.array(all_probs)
    y_true = np.array(all_true).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    return total_loss / total_seen, total_correct / total_seen, y_true, y_prob, y_pred



def save_artifacts(out_dir, history, y_true, y_prob, y_pred, test_loss, test_acc, bal_acc, roc_auc, ap):
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

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
    plt.savefig(out_dir / "training_curves.png", dpi=200)
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
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
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
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="Real")
    plt.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="Fake")
    plt.axvline(0.5, linestyle="--", linewidth=1.2, label="Threshold = 0.5")
    plt.xlabel("Predicted P(Fake)")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "score_distribution.png", dpi=200)
    plt.close()

    with open(out_dir / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"test_loss={test_loss:.6f}\n")
        f.write(f"test_acc={test_acc:.6f}\n")
        f.write(f"test_bal_acc={bal_acc:.6f}\n")
        f.write(f"roc_auc={roc_auc:.6f}\n")
        f.write(f"average_precision={ap:.6f}\n\n")
        f.write("classification_report\n")
        f.write(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))
        f.write("\nconfusion_matrix\n")
        f.write(str(cm))

    with open(out_dir / "test_predictions.csv", "w", encoding="utf-8") as f:
        f.write("true_label,prob_fake,pred_0.5\n")
        for yt, yp, yd in zip(y_true, y_prob, y_pred):
            f.write(f"{int(yt)},{float(yp):.8f},{int(yd)}\n")


def main():
    root_dir = detect_root()
    train_real_dir = find_class_dir(root_dir, "train", "real")
    train_fake_dir = find_class_dir(root_dir, "train", "fake")
    test_real_dir = find_class_dir(root_dir, "test", "real")
    test_fake_dir = find_class_dir(root_dir, "test", "fake")

    print(f"Device: {DEVICE}  |  AMP: {USE_AMP}")
    if torch.cuda.device_count() > 1:
        print(f"GPUs available: {torch.cuda.device_count()}")

    train_real_paths = get_image_paths(train_real_dir)
    train_fake_paths = get_image_paths(train_fake_dir)
    test_real_paths = get_image_paths(test_real_dir)
    test_fake_paths = get_image_paths(test_fake_dir)

    if len(train_real_paths) == 0 or len(train_fake_paths) == 0:
        raise RuntimeError(f"Training folders are empty or not found: {train_real_dir} | {train_fake_dir}")
    if len(test_real_paths) == 0 or len(test_fake_paths) == 0:
        raise RuntimeError(f"Test folders are empty or not found: {test_real_dir} | {test_fake_dir}")

    train_paths = train_real_paths + train_fake_paths
    train_labels = [0] * len(train_real_paths) + [1] * len(train_fake_paths)
    te_paths = test_real_paths + test_fake_paths
    te_labels = [0] * len(test_real_paths) + [1] * len(test_fake_paths)

    tr_paths, va_paths, tr_labels, va_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=VAL_SIZE,
        stratify=train_labels,
        random_state=SEED,
    )

    print(f"Dataset root: {root_dir.resolve()}")
    print(f"Train {len(tr_paths)} | Val {len(va_paths)} | Test {len(te_paths)}")

    train_loader = make_loader(tr_paths, tr_labels, train_transform, shuffle=True)
    val_loader = make_loader(va_paths, va_labels, eval_transform, shuffle=False)
    test_loader = make_loader(te_paths, te_labels, eval_transform, shuffle=False)

    print("\n" + "=" * 60)
    print("FINAL MODEL — B0 | 224px | AdamW | 5e-4 | partial | drop=0.2")
    print("=" * 60)

    model = build_model(dropout=0.2).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        print("-" * 78)
        print(f"Epoch {epoch:02d}/{EPOCHS} | lrs={[f'{lr:.7f}' for lr in current_lrs]}")

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, train=True, epoch_num=epoch, stage_name="TRAIN")
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, train=False, epoch_num=epoch, stage_name="VAL")
        scheduler.step(va_loss)

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(va_acc)

        print(
            f"[EPOCH SUMMARY] {epoch:02d}/{EPOCHS} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}",
            flush=True,
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            print(f"[CHECKPOINT] Saved new best model at epoch {epoch:02d}", flush=True)
        else:
            patience_count += 1
            print(f"[EARLY STOP WATCH] No improvement count = {patience_count}/{EARLY_STOP_PAT}", flush=True)
            if patience_count >= EARLY_STOP_PAT:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)
    print("\n[BEST MODEL RESTORED]")
    print(f"Best val loss        : {best_val_loss:.4f}")

    test_loss, test_acc, y_true, y_prob, y_pred = evaluate_test(model, test_loader, criterion)

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    print("\n[TEST EVALUATION]")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Test loss            : {test_loss:.4f}")
    print(f"Test acc             : {test_acc:.4f}")
    print(f"Test balanced acc    : {bal_acc:.4f}")
    print(f"ROC AUC              : {roc_auc:.4f}")
    print(f"Average precision    : {ap:.4f}")
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))
    print("[CONFUSION MATRIX]")
    print(cm)

    out_dir = Path("models/efficientnet/simple_logged_artifacts")
    save_artifacts(out_dir, history, y_true, y_prob, y_pred, test_loss, test_acc, bal_acc, roc_auc, ap)

    print(f"\nTraining curves saved to      : {(out_dir / 'training_curves.png').resolve()}")
    print(f"Confusion matrix image saved to: {(out_dir / 'confusion_matrix.png').resolve()}")
    print(f"ROC curve saved to            : {(out_dir / 'roc_curve.png').resolve()}")
    print(f"PR curve saved to             : {(out_dir / 'pr_curve.png').resolve()}")
    print(f"Score distribution saved to   : {(out_dir / 'score_distribution.png').resolve()}")
    print(f"Metrics summary saved to      : {(out_dir / 'metrics_summary.txt').resolve()}")
    print(f"Test predictions saved to     : {(out_dir / 'test_predictions.csv').resolve()}")
    print("\nDone ✓")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
