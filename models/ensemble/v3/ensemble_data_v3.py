import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DualTransformImageFolderSubset(Dataset):
    def __init__(self, subset, swin_tf, eff_tf):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.samples = self.dataset.samples
        self.targets = [self.dataset.targets[i] for i in self.indices]
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.swin_tf = swin_tf
        self.eff_tf = eff_tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        path, target = self.samples[sample_idx]
        img = Image.open(path).convert("RGB")
        swin_img = self.swin_tf(img)
        eff_img = self.eff_tf(img)
        return {"swin": swin_img, "efficientnet": eff_img}, target


class DualTransformImageFolder(Dataset):
    def __init__(self, root, swin_tf, eff_tf):
        self.base = ImageFolder(root)
        self.samples = self.base.samples
        self.targets = self.base.targets
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx
        self.swin_tf = swin_tf
        self.eff_tf = eff_tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        swin_img = self.swin_tf(img)
        eff_img = self.eff_tf(img)
        return {"swin": swin_img, "efficientnet": eff_img}, target


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(img_size, aug_cfg):
    color = aug_cfg.get("color_jitter", 0.0)
    erasing = aug_cfg.get("random_erasing", 0.0)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=aug_cfg.get("horizontal_flip", 0.5)),
        transforms.ColorJitter(
            brightness=color,
            contrast=color,
            saturation=color,
            hue=min(color / 2.0, 0.08),
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=erasing),
    ])


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloaders(cfg):
    eff_cfg = cfg["efficientnet"]
    swin_cfg = cfg["swin"]
    aug_cfg = cfg.get("augment", {})

    full_ds = ImageFolder(cfg["train_dir"])
    classes = full_ds.classes
    class_to_idx = full_ds.class_to_idx
    if "fake" not in class_to_idx or "real" not in class_to_idx:
        raise RuntimeError(f"Expected classes fake/real. Got: {classes} mapping={class_to_idx}")

    n_total = len(full_ds)
    n_val = int(n_total * cfg["val_ratio"])
    n_train = n_total - n_val

    base_train_ds, base_val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    train_swin_tf = build_train_transform(swin_cfg["img_size"], aug_cfg)
    train_eff_tf = build_train_transform(eff_cfg["img_size"], aug_cfg)
    eval_swin_tf = build_eval_transform(swin_cfg["img_size"])
    eval_eff_tf = build_eval_transform(eff_cfg["img_size"])

    train_ds = DualTransformImageFolderSubset(base_train_ds, train_swin_tf, train_eff_tf)
    val_ds = DualTransformImageFolderSubset(base_val_ds, eval_swin_tf, eval_eff_tf)
    test_ds = DualTransformImageFolder(cfg["test_dir"], eval_swin_tf, eval_eff_tf)

    if test_ds.class_to_idx != class_to_idx:
        raise RuntimeError(f"Class mapping mismatch. train={class_to_idx} test={test_ds.class_to_idx}")

    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "pin_memory": True,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "full_ds": full_ds,
        "base_train_ds": base_train_ds,
        "base_val_ds": base_val_ds,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
    }
