import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.swin.v1.config import CONFIG
from models.swin.v1.model_utils import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def find_best_threshold(y_true, probs, metric="balanced_acc"):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_s = -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        if metric == "balanced_acc":
            s = balanced_accuracy(y_true, pred)
        else:
            s = balanced_accuracy(y_true, pred)
        if s > best_s:
            best_s = s
            best_t = float(t)
    return best_t, best_s


def eval_on_loader(model, loader, device, threshold_metric):
    model.eval()
    all_logits = []
    all_y = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1).detach().cpu().numpy()  # (B,)
            all_logits.append(logits)
            all_y.append(y.numpy())

    all_logits = np.concatenate(all_logits)
    all_y = np.concatenate(all_y).astype(int)

    probs_fake = sigmoid(all_logits)  # interpret as P(fake=1)
    auc = None
    try:
        auc = roc_auc_score(all_y, probs_fake)
    except:
        auc = None

    best_t, best_score = find_best_threshold(all_y, probs_fake, metric=threshold_metric)
    pred = (probs_fake >= best_t).astype(int)

    pred_counts = np.bincount(pred, minlength=2)
    true_counts = np.bincount(all_y, minlength=2)

    return best_t, best_score, auc, true_counts, pred_counts


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    img_size = cfg["img_size"]
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    full_ds = ImageFolder(cfg["train_dir"], transform=train_tf)

    # Label mapping: fake=0, real=1 in your run.
    # We'll redefine for binary training: y=1 means FAKE, y=0 means REAL.
    # So we must remap targets accordingly.
    # If your mapping is {'fake':0,'real':1}, then y_fake = (target == 0).
    if full_ds.class_to_idx != {"fake": 0, "real": 1} and "fake" in full_ds.class_to_idx and "real" in full_ds.class_to_idx:
        pass

    n_total = len(full_ds)
    n_val = int(n_total * cfg["val_ratio"])
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )
    val_ds.dataset.transform = val_tf

    classes = full_ds.classes
    class_to_idx = full_ds.class_to_idx
    print("Classes:", classes)
    print("Mapping:", class_to_idx)
    print("Total train folder size:", n_total)
    print("Train split size:", len(train_ds))
    print("Val split size:", len(val_ds))

    # Build counts on TRAIN split with remapping (fake=1)
    train_targets_orig = np.array(full_ds.targets)[train_ds.indices]  # 0=fake, 1=real
    train_targets_bin = (train_targets_orig == class_to_idx.get("fake", 0)).astype(int)  # fake->1, real->0
    fake_count = int(train_targets_bin.sum())
    real_count = int(len(train_targets_bin) - fake_count)
    print("Train split counts (real=0, fake=1):", [real_count, fake_count])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )

    model = build_model(cfg["model_name"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    # pos_weight is weight for positive class (fake=1)
    pos_weight = torch.tensor([real_count / (fake_count + 1e-12)], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("Using BCEWithLogitsLoss pos_weight (fake=1):", float(pos_weight.item()))

    best_val_score = -1.0
    best_path = os.path.join(cfg["checkpoint_dir"], "best.pt")

    for epoch in range(cfg["epochs"]):
        model.train()
        losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} Train"):
            x = x.to(device, non_blocking=True)
            # remap y: fake->1, real->0
            y_bin = (y == class_to_idx.get("fake", 0)).float().to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)  # (B,)
            loss = loss_fn(logits, y_bin)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        best_t, val_score, val_auc, true_counts, pred_counts = eval_on_loader(
            model, val_loader, device, cfg["threshold_metric"]
        )

        msg = f"epoch={epoch+1} train_loss={float(np.mean(losses)):.4f} val_{cfg['threshold_metric']}={val_score:.4f} best_t={best_t:.2f}"
        if val_auc is not None:
            msg += f" val_auc={val_auc:.4f}"
        print(msg)
        print("Val true counts (real=0,fake=1):", true_counts, "Val pred counts:", pred_counts)

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "best_threshold": best_t,
                    "val_score": val_score,
                    "val_auc": val_auc,
                },
                best_path
            )
            print("Saved best checkpoint:", best_path)


if __name__ == "__main__":
    main()