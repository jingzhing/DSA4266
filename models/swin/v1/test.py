import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

from models.swin.v1.config import CONFIG
from models.swin.v1.model_utils import build_model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def balanced_accuracy(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    return 0.5 * (tpr + tnr)


def main():
    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}. Train first.")

    img_size = cfg["img_size"]
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    test_ds = ImageFolder(cfg["test_dir"], transform=test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_t = float(ckpt.get("best_threshold", cfg["default_threshold"]))
    classes = ckpt.get("classes", test_ds.classes)
    class_to_idx = ckpt.get("class_to_idx", test_ds.class_to_idx)

    if class_to_idx != test_ds.class_to_idx:
        raise RuntimeError(f"Class mapping mismatch. ckpt={class_to_idx} test={test_ds.class_to_idx}")

    model = build_model(cfg["model_name"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_logits = []
    all_y = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1).detach().cpu().numpy()
            all_logits.append(logits)
            all_y.append(y.numpy())

    all_logits = np.concatenate(all_logits)
    y_orig = np.concatenate(all_y).astype(int)  # 0=fake,1=real in your mapping

    # remap to binary y: fake=1, real=0
    y_bin = (y_orig == class_to_idx.get("fake", 0)).astype(int)

    probs_fake = sigmoid(all_logits)
    pred_bin = (probs_fake >= best_t).astype(int)

    auc = None
    try:
        auc = roc_auc_score(y_bin, probs_fake)
    except:
        auc = None

    bal_acc = balanced_accuracy(y_bin, pred_bin)
    acc = accuracy_score(y_bin, pred_bin)

    true_counts = np.bincount(y_bin, minlength=2)
    pred_counts = np.bincount(pred_bin, minlength=2)

    cm = confusion_matrix(y_bin, pred_bin)
    report = classification_report(y_bin, pred_bin, target_names=["real(0)", "fake(1)"], digits=4, zero_division=0)

    print("Classes:", classes)
    print("Mapping:", class_to_idx)
    print("Using threshold:", best_t)
    print("True counts (real=0,fake=1):", true_counts, "Total:", true_counts.sum())
    print("Pred counts (real=0,fake=1):", pred_counts, "Total:", pred_counts.sum())
    print(f"Accuracy (on y_bin): {acc:.4f}")
    print(f"Balanced Acc: {bal_acc:.4f}")
    if auc is not None:
        print(f"AUC (fake=1): {auc:.4f}")
    print("Confusion Matrix (rows true [real,fake], cols pred [real,fake]):")
    print(cm)
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    main()