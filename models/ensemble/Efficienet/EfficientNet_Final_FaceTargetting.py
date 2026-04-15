
import os
import random
import warnings
from pathlib import Path

import cv2
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

SEED = 42
BATCH_SIZE = 128
EPOCHS = 8
EARLY_STOP_PAT = 5
IMG_SIZE = 224
NUM_WORKERS = 2
VAL_SIZE = 0.2

USE_FACE_CROP = True
FACE_MARGIN = 0.25
FACE_MIN_SIZE = 40
SAVE_FACE_PREVIEW_COUNT = 24

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

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = None


def log(msg=""):
    print(msg, flush=True)


def get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
    return _FACE_CASCADE


def detect_and_crop_face(pil_img, margin=0.25, min_size=40):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
    )

    if faces is None or len(faces) == 0:
        return pil_img, False

    areas = [w * h for (x, y, w, h) in faces]
    x, y, w, h = faces[int(np.argmax(areas))]

    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img.shape[1], x + w + mx)
    y2 = min(img.shape[0], y + h + my)

    if x2 <= x1 or y2 <= y1:
        return pil_img, False

    cropped = pil_img.crop((x1, y1, x2, y2))
    return cropped, True


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
        "Could not find dataset root. Set DEEPNET_ROOT or place your data under a standard structure like "
        "data/deepdetect-2025_dddata/train/{real,fake} and test/{real,fake}."
    )


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
    def __init__(self, paths, labels, transform, use_face_crop=False, face_margin=0.25, face_min_size=40):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.use_face_crop = use_face_crop
        self.face_margin = face_margin
        self.face_min_size = face_min_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            face_found = False
            if self.use_face_crop:
                img, face_found = detect_and_crop_face(
                    img,
                    margin=self.face_margin,
                    min_size=self.face_min_size,
                )
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            face_found = False

        return img, torch.tensor(self.labels[idx], dtype=torch.float32), str(path), int(face_found)


def make_loader(paths, labels, transform, shuffle, use_face_crop):
    ds = DeepfakeDataset(
        paths,
        labels,
        transform,
        use_face_crop=use_face_crop,
        face_margin=FACE_MARGIN,
        face_min_size=FACE_MIN_SIZE,
    )
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


def build_model(dropout=0.2):
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


def run_epoch(model, loader, criterion, optimizer, scaler, train=True, epoch_num=None, stage_name=None):
    stage_name = stage_name or ("TRAIN" if train else "VAL")
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    total_faces = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = f"Epoch {epoch_num:02d} {stage_name}" if epoch_num is not None else stage_name

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch_idx, (imgs, labels, paths, face_found) in enumerate(pbar, start=1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            total_faces += int(face_found.sum().item())

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
            running_face_rate = total_faces / total_seen
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}", face=f"{running_face_rate:.3f}")

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                epoch_part = f"epoch={epoch_num:02d} " if epoch_num is not None else ""
                print(
                    f"[{stage_name}] {epoch_part}batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={running_loss:.4f} acc={running_acc:.4f} face_rate={running_face_rate:.3f}",
                    flush=True,
                )

    return total_loss / total_seen, total_correct / total_seen, total_faces / max(total_seen, 1)


def evaluate_test(model, loader, criterion):
    model.eval()
    all_probs, all_true, all_paths = [], [], []
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    total_faces = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="TEST", leave=False, dynamic_ncols=True)
        for batch_idx, (imgs, labels, paths, face_found) in enumerate(pbar, start=1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            total_faces += int(face_found.sum().item())

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
            all_paths.extend(paths)

            running_loss = total_loss / total_seen
            running_acc = total_correct / total_seen
            running_face_rate = total_faces / total_seen
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}", face=f"{running_face_rate:.3f}")

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                print(
                    f"[TEST] batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={running_loss:.4f} acc={running_acc:.4f} face_rate={running_face_rate:.3f}",
                    flush=True,
                )

    y_prob = np.array(all_probs)
    y_true = np.array(all_true).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    return (
        total_loss / total_seen,
        total_correct / total_seen,
        total_faces / max(total_seen, 1),
        y_true,
        y_prob,
        y_pred,
        all_paths,
    )


def save_face_preview(paths, out_dir, limit=24):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, path in enumerate(paths[: max(limit * 3, limit)]):
        try:
            img = Image.open(path).convert("RGB")
            cropped, found = detect_and_crop_face(img, margin=FACE_MARGIN, min_size=FACE_MIN_SIZE)
            if not found:
                continue
            preview = Image.new("RGB", (img.width + cropped.width, max(img.height, cropped.height)), (255, 255, 255))
            preview.paste(img, (0, 0))
            preview.paste(cropped, (img.width, 0))
            preview.save(out_dir / f"face_preview_{saved:03d}.png")
            saved += 1
            if saved >= limit:
                break
        except Exception:
            continue
    return saved


def main():
    root_dir = detect_root()
    train_real_dir = find_class_dir(root_dir, "train", "real")
    train_fake_dir = find_class_dir(root_dir, "train", "fake")
    test_real_dir = find_class_dir(root_dir, "test", "real")
    test_fake_dir = find_class_dir(root_dir, "test", "fake")

    print(f"Device: {DEVICE}  |  AMP: {USE_AMP}")
    print(f"Face crop enabled: {USE_FACE_CROP} | margin={FACE_MARGIN} | min_size={FACE_MIN_SIZE}")
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

    train_loader = make_loader(tr_paths, tr_labels, train_transform, shuffle=True, use_face_crop=USE_FACE_CROP)
    val_loader = make_loader(va_paths, va_labels, eval_transform, shuffle=False, use_face_crop=USE_FACE_CROP)
    test_loader = make_loader(te_paths, te_labels, eval_transform, shuffle=False, use_face_crop=USE_FACE_CROP)

    print("\n" + "=" * 72)
    print("FACE-CROP MODEL — B0 | 224px | AdamW | 5e-4 | partial | drop=0.2")
    print("=" * 72)

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

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "train_face_rate": [], "val_face_rate": []}
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        print("-" * 78)
        print(f"Epoch {epoch:02d}/{EPOCHS} | lrs={[f'{lr:.7f}' for lr in current_lrs]}")

        tr_loss, tr_acc, tr_face_rate = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            train=True, epoch_num=epoch, stage_name="TRAIN"
        )
        va_loss, va_acc, va_face_rate = run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            train=False, epoch_num=epoch, stage_name="VAL"
        )
        scheduler.step(va_loss)

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(va_acc)
        history["train_face_rate"].append(tr_face_rate)
        history["val_face_rate"].append(va_face_rate)

        print(
            f"[EPOCH SUMMARY] {epoch:02d}/{EPOCHS} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_face_rate={tr_face_rate:.3f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_face_rate={va_face_rate:.3f}",
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

    test_loss, test_acc, test_face_rate, y_true, y_prob, y_pred, test_paths_out = evaluate_test(model, test_loader, criterion)

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print("\n[TEST EVALUATION]")
    print(f"Test loss            : {test_loss:.4f}")
    print(f"Test acc             : {test_acc:.4f}")
    print(f"Test balanced acc    : {bal_acc:.4f}")
    print(f"ROC AUC              : {roc_auc:.4f}")
    print(f"Average precision    : {ap:.4f}")
    print(f"Test face rate       : {test_face_rate:.3f}")
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))
    print("[CONFUSION MATRIX]")
    print(cm)

    out_dir = Path("models/efficientnet/facecrop_logged_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    preview_saved = save_face_preview(tr_paths, out_dir / "face_previews", limit=SAVE_FACE_PREVIEW_COUNT)

    with open(out_dir / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"use_face_crop={USE_FACE_CROP}\n")
        f.write(f"face_margin={FACE_MARGIN}\n")
        f.write(f"face_min_size={FACE_MIN_SIZE}\n")
        f.write(f"test_loss={test_loss:.6f}\n")
        f.write(f"test_acc={test_acc:.6f}\n")
        f.write(f"test_bal_acc={bal_acc:.6f}\n")
        f.write(f"roc_auc={roc_auc:.6f}\n")
        f.write(f"average_precision={ap:.6f}\n")
        f.write(f"test_face_rate={test_face_rate:.6f}\n")
        f.write(f"face_preview_saved={preview_saved}\n\n")
        f.write("classification_report\n")
        f.write(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))
        f.write("\nconfusion_matrix\n")
        f.write(str(cm))

    with open(out_dir / "test_predictions.csv", "w", encoding="utf-8") as f:
        f.write("path,true_label,prob_fake,pred_0.5\n")
        for p, yt, yp, yd in zip(test_paths_out, y_true, y_prob, y_pred):
            f.write(f"{p},{int(yt)},{float(yp):.8f},{int(yd)}\n")

    print(f"\nConfusion matrix image saved to: {(out_dir / 'confusion_matrix.png').resolve()}")
    print(f"Metrics summary saved to      : {(out_dir / 'metrics_summary.txt').resolve()}")
    print(f"Face preview folder           : {(out_dir / 'face_previews').resolve()} (saved {preview_saved})")
    print("\nDone ✓")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
