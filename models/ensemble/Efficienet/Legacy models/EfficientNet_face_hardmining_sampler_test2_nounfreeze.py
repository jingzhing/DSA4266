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
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

warnings.filterwarnings("ignore")

SEED = 42
BATCH_SIZE = 128
EPOCHS = 8
EARLY_STOP_PAT = 3
IMG_SIZE = 224
NUM_WORKERS = 4
TEST_NUM_WORKERS = 2
VAL_SIZE = 0.2
DROPOUT = 0.25
BACKBONE_LR = 5e-5
HEAD_LR = 3e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
UNFREEZE_EPOCH = None

FACE_REGION_ENABLED = True
FACE_MARGIN = 0.35
FACE_DETECT_MAX_SIDE = 640
FACE_DETECT_SCALE_FACTOR = 1.1
FACE_DETECT_MIN_NEIGHBORS = 5
FACE_DETECT_MIN_SIZE = 40
FACE_FALLBACK_TO_ORIGINAL = True

HARD_FALSE_POSITIVE_MINING = True
HARD_MINING_START_EPOCH = 2
HARD_MINING_TOPK = 3000
HARD_MINING_MIN_PROB = 0.50
HARD_MINING_MAX_FRAC_REAL = 0.20
HARD_MINING_EXTRA_COPIES = 1

CHECKPOINT_DIR = Path("models/efficientnet/face_hardmining_run")
CHECKPOINT_NAME = "best_efficientnet_b0_face_hardmining.pt"
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
_FACE_CASCADE = None
_FACE_BACKEND_MSG = None


def log(msg=""):
    print(msg, flush=True)


def get_face_cascade():
    global _FACE_CASCADE, _FACE_BACKEND_MSG

    if not FACE_REGION_ENABLED:
        if _FACE_BACKEND_MSG is None:
            _FACE_BACKEND_MSG = "[FACE] Disabled"
        return None

    if not CV2_AVAILABLE:
        if _FACE_BACKEND_MSG is None:
            _FACE_BACKEND_MSG = "[FACE] OpenCV not available, using original images"
            log(_FACE_BACKEND_MSG)
        return None

    if _FACE_CASCADE is not None:
        return _FACE_CASCADE

    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            if _FACE_BACKEND_MSG is None:
                _FACE_BACKEND_MSG = f"[FACE] Haar cascade not found at {cascade_path}, using original images"
                log(_FACE_BACKEND_MSG)
            return None
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            if _FACE_BACKEND_MSG is None:
                _FACE_BACKEND_MSG = "[FACE] Failed to load Haar cascade, using original images"
                log(_FACE_BACKEND_MSG)
            return None
        _FACE_CASCADE = cascade
        if _FACE_BACKEND_MSG is None:
            _FACE_BACKEND_MSG = f"[FACE] Using OpenCV Haar cascade: {cascade_path}"
            log(_FACE_BACKEND_MSG)
        return _FACE_CASCADE
    except Exception as e:
        if _FACE_BACKEND_MSG is None:
            _FACE_BACKEND_MSG = f"[FACE] Face detector init failed: {e}. Using original images"
            log(_FACE_BACKEND_MSG)
        return None


def crop_largest_face(pil_img, cache=None, cache_key=None):
    if not FACE_REGION_ENABLED:
        return pil_img, False

    if cache is not None and cache_key in cache:
        box = cache[cache_key]
        if box is None:
            return pil_img, False
        return pil_img.crop(box), True

    cascade = get_face_cascade()
    if cascade is None:
        if cache is not None and cache_key is not None:
            cache[cache_key] = None
        return pil_img, False

    rgb = np.array(pil_img)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        if cache is not None and cache_key is not None:
            cache[cache_key] = None
        return pil_img, False

    h, w = rgb.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    detect_img = rgb
    if max_side > FACE_DETECT_MAX_SIDE:
        scale = FACE_DETECT_MAX_SIDE / max_side
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        detect_img = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(detect_img, cv2.COLOR_RGB2GRAY)
    min_size = max(16, int(round(FACE_DETECT_MIN_SIZE * scale)))

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECT_SCALE_FACTOR,
        minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
        minSize=(min_size, min_size),
    )

    if len(faces) == 0:
        if cache is not None and cache_key is not None:
            cache[cache_key] = None
        return pil_img, False

    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    if scale != 1.0:
        inv = 1.0 / scale
        x = int(round(x * inv))
        y = int(round(y * inv))
        fw = int(round(fw * inv))
        fh = int(round(fh * inv))

    mx = int(round(fw * FACE_MARGIN))
    my = int(round(fh * FACE_MARGIN))
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + fw + mx)
    y2 = min(h, y + fh + my)

    if x2 <= x1 or y2 <= y1:
        if cache is not None and cache_key is not None:
            cache[cache_key] = None
        return pil_img, False

    box = (x1, y1, x2, y2)
    if cache is not None and cache_key is not None:
        cache[cache_key] = box
    return pil_img.crop(box), True


class DeepfakeDataset(Dataset):
    def __init__(self, paths, labels, transform, use_face_region=False):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
        self.use_face_region = use_face_region
        self.face_cache = {}
        self.face_hits = 0
        self.face_misses = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.use_face_region:
                face_img, found = crop_largest_face(img, cache=self.face_cache, cache_key=path)
                if found:
                    self.face_hits += 1
                    img = face_img
                elif FACE_FALLBACK_TO_ORIGINAL:
                    self.face_misses += 1
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


class MutableWeightedSampler(Sampler):
    def __init__(self, num_items, weights=None, num_samples=None, replacement=True):
        self.num_items = int(num_items)
        self.replacement = replacement
        self.weights = torch.ones(self.num_items, dtype=torch.double) if weights is None else torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples) if num_samples is not None else self.num_items

    def set_weights(self, weights, num_samples=None):
        weights = torch.as_tensor(weights, dtype=torch.double)
        if weights.numel() != self.num_items:
            raise ValueError(f"weights length {weights.numel()} does not match sampler size {self.num_items}")
        self.weights = weights
        if num_samples is not None:
            self.num_samples = int(num_samples)

    def __iter__(self):
        sample_idxs = torch.multinomial(self.weights, self.num_samples, self.replacement)
        return iter(sample_idxs.tolist())

    def __len__(self):
        return self.num_samples


def build_sample_weights(base_paths, base_labels, hard_real_examples, extra_copies=1):
    weights = np.ones(len(base_paths), dtype=np.float64)
    if not HARD_FALSE_POSITIVE_MINING or not hard_real_examples or extra_copies <= 0:
        return weights, len(base_paths)

    index_by_path = {}
    for idx, path in enumerate(base_paths):
        if int(base_labels[idx]) == 0:
            index_by_path.setdefault(path, []).append(idx)

    selected_count = 0
    for path, _prob in hard_real_examples:
        for idx in index_by_path.get(path, []):
            weights[idx] += float(extra_copies)
            selected_count += 1

    num_samples = len(base_paths) + max(0, selected_count * int(extra_copies))
    return weights, num_samples


def collect_hard_real_examples(paths, y_true, y_prob, epoch, output_dir):
    if not HARD_FALSE_POSITIVE_MINING or epoch < HARD_MINING_START_EPOCH:
        return []

    real_scores = {}
    for path, yt, yp in zip(paths, y_true, y_prob):
        if int(yt) != 0:
            continue
        prob = float(yp)
        if path not in real_scores or prob > real_scores[path]:
            real_scores[path] = prob

    if not real_scores:
        return []

    ranked = sorted(real_scores.items(), key=lambda x: x[1], reverse=True)
    min_keep = [item for item in ranked if item[1] >= HARD_MINING_MIN_PROB]
    cap = max(1, int(round(len(real_scores) * HARD_MINING_MAX_FRAC_REAL)))
    keep_n = min(HARD_MINING_TOPK, cap)

    if len(min_keep) >= keep_n:
        hard_examples = min_keep[:keep_n]
    else:
        hard_examples = ranked[:keep_n]

    csv_path = output_dir / f"hard_real_epoch_{epoch:02d}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "prob_fake"])
        for path, prob in hard_examples:
            writer.writerow([path, f"{prob:.8f}"])

    return hard_examples


def make_loader(paths, labels, transform, shuffle, loader_name, use_face_region=False, sampler=None, num_workers=None, persistent_workers=None):
    ds = DeepfakeDataset(paths, labels, transform, use_face_region=use_face_region)
    loader_num_workers = NUM_WORKERS if num_workers is None else int(num_workers)
    if persistent_workers is None:
        loader_persistent_workers = loader_num_workers > 0
    else:
        loader_persistent_workers = bool(persistent_workers) and loader_num_workers > 0

    loader_kwargs = dict(
        dataset=ds,
        batch_size=BATCH_SIZE,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=loader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if loader_num_workers > 0:
        loader_kwargs["persistent_workers"] = loader_persistent_workers
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)
    sampler_desc = "weighted_sampler" if sampler is not None else ("shuffle" if shuffle else "sequential")
    log(
        f"[LOADER] {loader_name:<10} batches={len(loader)} samples={len(ds)} "
        f"mode={sampler_desc} face_region={use_face_region} workers={loader_num_workers}"
    )
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
        patience=3,
        min_lr=1e-7,
    )


def maybe_unfreeze_all(model, optimizer, scheduler, current_epoch):
    if UNFREEZE_EPOCH is None or current_epoch != UNFREEZE_EPOCH:
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


def search_best_threshold(y_true, y_prob, start=0.40, end=0.80, step=0.02):
    best_row = None
    thr = start
    while thr <= end + 1e-12:
        thr = round(thr, 2)
        y_pred = (y_prob >= thr).astype(int)
        row = {
            "threshold": thr,
            "acc": float((y_pred == y_true).mean()),
            "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "cm": confusion_matrix(y_true, y_pred).tolist(),
        }
        if (best_row is None or row["bal_acc"] > best_row["bal_acc"] or
            (row["bal_acc"] == best_row["bal_acc"] and row["acc"] > best_row["acc"])):
            best_row = row
        thr += step
    return best_row


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
    log("EFFICIENTNET B0 FACE-REGION + HARD-REAL-MINING RUN")
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
    log(f"Test workers         : {TEST_NUM_WORKERS}")
    log(f"Checkpoint           : {CHECKPOINT_DIR / CHECKPOINT_NAME}")
    log(f"Face region enabled  : {FACE_REGION_ENABLED}")
    log(f"Hard mining enabled  : {HARD_FALSE_POSITIVE_MINING}")
    if HARD_FALSE_POSITIVE_MINING:
        log(f"Hard mining start    : epoch {HARD_MINING_START_EPOCH}")
        log(f"Hard mining topk     : {HARD_MINING_TOPK}")
        log(f"Hard mining min prob : {HARD_MINING_MIN_PROB}")
        log(f"Hard mining copies   : +{HARD_MINING_EXTRA_COPIES}")
    if torch.cuda.device_count() > 1:
        log(f"GPUs available       : {torch.cuda.device_count()}")

    get_face_cascade()

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

    TR_PATHS = list(TR_PATHS)
    TR_LABELS = list(TR_LABELS)
    VA_PATHS = list(VA_PATHS)
    VA_LABELS = list(VA_LABELS)

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
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.95, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_sampler = MutableWeightedSampler(len(TR_PATHS))
    train_loader = make_loader(
        TR_PATHS,
        TR_LABELS,
        train_transform,
        False,
        "train",
        use_face_region=FACE_REGION_ENABLED,
        sampler=train_sampler,
    )
    val_loader = make_loader(VA_PATHS, VA_LABELS, eval_transform, False, "val", use_face_region=FACE_REGION_ENABLED)
    test_loader = make_loader(
        test_paths,
        test_labels,
        eval_transform,
        False,
        "test",
        use_face_region=FACE_REGION_ENABLED,
        num_workers=TEST_NUM_WORKERS,
        persistent_workers=False,
    )

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
    log("Threshold search     : 0.40 to 0.80 step 0.02")
    log(f"Unfreeze epoch       : {UNFREEZE_EPOCH if UNFREEZE_EPOCH is not None else 'disabled'}")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    patience_count = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_dir = CHECKPOINT_DIR / ARTIFACTS_DIRNAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME

    current_hard_real_examples = []

    log("\n[TRAINING START]")
    for epoch in range(1, EPOCHS + 1):
        optimizer, scheduler = maybe_unfreeze_all(model, optimizer, scheduler, epoch)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        sample_weights, sample_count = build_sample_weights(
            TR_PATHS,
            TR_LABELS,
            current_hard_real_examples,
            extra_copies=HARD_MINING_EXTRA_COPIES,
        )
        train_sampler.set_weights(sample_weights, num_samples=sample_count)
        log("-" * 80)
        log(f"Epoch {epoch:02d}/{EPOCHS} | lrs={[f'{lr:.7f}' for lr in current_lrs]}")
        if current_hard_real_examples:
            log(
                f"[HARD MINING] upweighted hard real samples this epoch: {len(current_hard_real_examples)} "
                f"| sampled_examples={sample_count}"
            )

        tr_loss, tr_acc, tr_bal_acc, tr_y_true, tr_y_prob, _, tr_paths_out = run_epoch(
            train_loader, model, criterion, optimizer, scaler,
            train=True, epoch_num=epoch, stage_name="TRAIN", threshold=THRESHOLD
        )
        va_loss, va_acc, va_bal_acc, va_y_true, va_y_prob, _, _ = run_epoch(
            val_loader, model, criterion, optimizer, scaler,
            train=False, epoch_num=epoch, stage_name="VAL", threshold=THRESHOLD
        )
        val_best_thr = search_best_threshold(va_y_true, va_y_prob)
        log(
            f"[VAL THRESHOLD SEARCH] best_thr={val_best_thr['threshold']:.2f} "
            f"acc={val_best_thr['acc']:.4f} bal_acc={val_best_thr['bal_acc']:.4f} cm={val_best_thr['cm']}"
        )

        current_hard_real_examples = collect_hard_real_examples(
            tr_paths_out,
            tr_y_true,
            tr_y_prob,
            epoch,
            artifacts_dir,
        )
        if current_hard_real_examples:
            top_preview = current_hard_real_examples[:3]
            log(f"[HARD MINING] selected {len(current_hard_real_examples)} hard real samples for next epoch")
            for idx, (hp, hprob) in enumerate(top_preview, start=1):
                log(f"    hard_real_{idx}: prob_fake={hprob:.4f} | {hp}")

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
                    "best_val_threshold": val_best_thr["threshold"],
                    "config": {
                        "img_size": IMG_SIZE,
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "backbone_lr": BACKBONE_LR,
                        "head_lr": HEAD_LR,
                        "dropout": DROPOUT,
                        "threshold": THRESHOLD,
                        "best_val_threshold": val_best_thr["threshold"],
                        "seed": SEED,
                        "root_dir": str(ROOT_DIR),
                        "num_workers": NUM_WORKERS,
                        "unfreeze_epoch": UNFREEZE_EPOCH,
                        "face_region_enabled": FACE_REGION_ENABLED,
                        "hard_false_positive_mining": HARD_FALSE_POSITIVE_MINING,
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
    best_threshold = float(ckpt.get("best_val_threshold", THRESHOLD))
    log("\n[BEST MODEL RESTORED]")
    log(f"Best epoch           : {ckpt['epoch']}")
    log(f"Best val loss        : {ckpt['best_val_loss']:.4f}")
    log(f"Best val threshold   : {best_threshold:.2f}")

    log("\n[TEST EVALUATION]")
    test_loss, test_acc, test_bal_acc, y_true, y_prob, y_pred, test_paths_out = run_epoch(
        test_loader, model, criterion, optimizer, scaler,
        train=False, epoch_num=None, stage_name="TEST", threshold=best_threshold
    )

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    report_text = classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)
    thr_rows = threshold_diagnostics(y_true, y_prob)
    searched_thr = search_best_threshold(y_true, y_prob)

    log(f"Test loss            : {test_loss:.4f}")
    log(f"Test acc             : {test_acc:.4f}")
    log(f"Test balanced acc    : {test_bal_acc:.4f}")
    log(f"ROC AUC              : {roc_auc:.4f}")
    log(f"Average precision    : {ap:.4f}")
    log(f"Applied threshold    : {best_threshold:.2f}")
    log(
        f"Test searched thr    : {searched_thr['threshold']:.2f} | "
        f"acc={searched_thr['acc']:.4f} | bal_acc={searched_thr['bal_acc']:.4f} | cm={searched_thr['cm']}"
    )
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

    save_artifacts(artifacts_dir, history, y_true, y_prob, y_pred, best_threshold, report_text, thr_rows)

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
        f.write(f"test_num_workers={TEST_NUM_WORKERS}\n")
        f.write(f"val_size={VAL_SIZE}\n")
        f.write(f"dropout={DROPOUT}\n")
        f.write(f"backbone_lr={BACKBONE_LR}\n")
        f.write(f"head_lr={HEAD_LR}\n")
        f.write(f"weight_decay={WEIGHT_DECAY}\n")
        f.write(f"threshold={THRESHOLD}\n")
        f.write(f"threshold_search_start=0.40\n")
        f.write(f"threshold_search_end=0.80\n")
        f.write(f"threshold_search_step=0.02\n")
        f.write(f"unfreeze_epoch={UNFREEZE_EPOCH if UNFREEZE_EPOCH is not None else 'disabled'}\n")
        f.write(f"checkpoint_dir={CHECKPOINT_DIR}\n")
        f.write(f"checkpoint_name={CHECKPOINT_NAME}\n")
        f.write(f"dataset_root={ROOT_DIR}\n")
        f.write(f"face_region_enabled={FACE_REGION_ENABLED}\n")
        f.write(f"hard_false_positive_mining={HARD_FALSE_POSITIVE_MINING}\n")
        f.write("train_loader_strategy=stable_dataloader_with_mutable_weighted_sampler\n")
        f.write(f"face_region_enabled={FACE_REGION_ENABLED}\n")
        f.write(f"face_margin={FACE_MARGIN}\n")
        f.write(f"hard_false_positive_mining={HARD_FALSE_POSITIVE_MINING}\n")
        f.write(f"hard_mining_start_epoch={HARD_MINING_START_EPOCH}\n")
        f.write(f"hard_mining_topk={HARD_MINING_TOPK}\n")
        f.write(f"hard_mining_min_prob={HARD_MINING_MIN_PROB}\n")
        f.write(f"hard_mining_max_frac_real={HARD_MINING_MAX_FRAC_REAL}\n")
        f.write(f"hard_mining_extra_copies={HARD_MINING_EXTRA_COPIES}\n")

    with open(artifacts_dir / "test_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "prob_fake", f"pred_{best_threshold:.2f}"])
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
