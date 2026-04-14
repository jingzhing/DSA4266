import json
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TransformSubset(Dataset):
    def __init__(self, subset, transform, include_path=False):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.samples = self.dataset.samples
        self.targets = [self.dataset.targets[i] for i in self.indices]
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.transform = transform
        self.include_path = include_path

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        path, target = self.samples[sample_idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        if self.include_path:
            return img, target, path
        return img, target


class TransformImageFolder(Dataset):
    def __init__(self, dataset, transform, include_path=False):
        self.dataset = dataset
        self.samples = dataset.samples
        self.targets = dataset.targets
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.transform = transform
        self.include_path = include_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        if self.include_path:
            return img, target, path
        return img, target

class Multiply255:
    def __call__(self, x):
        return x * 255.0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(img_size, aug_cfg):
    color = aug_cfg.get("color_jitter", 0.0)
    erasing = aug_cfg.get("random_erasing", 0.0)
    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=aug_cfg.get("horizontal_flip", 0.5)),
    ]
    if aug_cfg.get("vertical_flip", 0.0) > 0:
        ops.append(transforms.RandomVerticalFlip(p=aug_cfg["vertical_flip"]))
    if color > 0:
        ops.append(
            transforms.ColorJitter(
                brightness=color,
                contrast=color,
                saturation=color,
                hue=min(color / 2.0, 0.08),
            )
        )
    ops.extend([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        Multiply255(),
    ])

    if erasing > 0:
        ops.append(transforms.RandomErasing(p=erasing))
    return transforms.Compose(ops)


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        Multiply255(),
    ])

    
def stratified_three_way_split_indices(targets, val_select_ratio, val_tune_ratio, seed):
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_indices = []
    val_select_indices = []
    val_tune_indices = []

    for cls in np.unique(targets):
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)

        n_total = len(cls_indices)
        n_val_select = max(1, int(round(n_total * val_select_ratio)))
        n_val_tune = max(1, int(round(n_total * val_tune_ratio)))

        if n_val_select + n_val_tune >= n_total:
            raise RuntimeError(
                f"Split ratios leave no training samples for class {cls}. "
                f"class_count={n_total}, val_select={n_val_select}, val_tune={n_val_tune}"
            )

        val_select_cls = cls_indices[:n_val_select]
        val_tune_cls = cls_indices[n_val_select:n_val_select + n_val_tune]
        train_cls = cls_indices[n_val_select + n_val_tune:]

        val_select_indices.extend(val_select_cls.tolist())
        val_tune_indices.extend(val_tune_cls.tolist())
        train_indices.extend(train_cls.tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_select_indices)
    rng.shuffle(val_tune_indices)
    return train_indices, val_select_indices, val_tune_indices


def load_or_create_fixed_split(full_ds, cfg):
    split_path = cfg.get("split_path")
    val_select_ratio = cfg["val_select_ratio"]
    val_tune_ratio = cfg["val_tune_ratio"]

    if split_path is None:
        return stratified_three_way_split_indices(
            full_ds.targets,
            val_select_ratio,
            val_tune_ratio,
            cfg["seed"],
        )

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        train_indices = payload["train_indices"]
        val_select_indices = payload["val_select_indices"]
        val_tune_indices = payload["val_tune_indices"]
        total = len(train_indices) + len(val_select_indices) + len(val_tune_indices)
        if total != len(full_ds):
            raise RuntimeError("Saved split does not match current dataset size")
        return train_indices, val_select_indices, val_tune_indices

    train_indices, val_select_indices, val_tune_indices = stratified_three_way_split_indices(
        full_ds.targets,
        val_select_ratio,
        val_tune_ratio,
        cfg["seed"],
    )
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": cfg["seed"],
            "val_select_ratio": val_select_ratio,
            "val_tune_ratio": val_tune_ratio,
            "dataset_size": len(full_ds),
            "train_indices": train_indices,
            "val_select_indices": val_select_indices,
            "val_tune_indices": val_tune_indices,
        }, f, indent=2)
    return train_indices, val_select_indices, val_tune_indices


def build_weighted_sampler(targets):
    targets = np.asarray(targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_single_model_dataloaders(cfg):
    model_cfg = cfg["model"]
    aug_cfg = cfg.get("augment", {})

    full_ds = ImageFolder(cfg["train_dir"])
    classes = full_ds.classes
    class_to_idx = full_ds.class_to_idx
    if "fake" not in class_to_idx or "real" not in class_to_idx:
        raise RuntimeError(f"Expected classes fake/real. Got: {classes} mapping={class_to_idx}")

    train_indices, val_select_indices, val_tune_indices = load_or_create_fixed_split(full_ds, cfg)

    base_train_ds = Subset(full_ds, train_indices)
    base_train_ds.indices = train_indices
    base_val_select_ds = Subset(full_ds, val_select_indices)
    base_val_select_ds.indices = val_select_indices
    base_val_tune_ds = Subset(full_ds, val_tune_indices)
    base_val_tune_ds.indices = val_tune_indices

    train_tf = build_train_transform(model_cfg["img_size"], aug_cfg)
    eval_tf = build_eval_transform(model_cfg["img_size"])

    train_ds = TransformSubset(base_train_ds, train_tf, include_path=False)
    val_select_ds = TransformSubset(base_val_select_ds, eval_tf, include_path=True)
    val_tune_ds = TransformSubset(base_val_tune_ds, eval_tf, include_path=True)
    test_base = ImageFolder(cfg["test_dir"])
    test_ds = TransformImageFolder(test_base, eval_tf, include_path=True)

    sampler = None
    shuffle = True
    if cfg.get("data_balance", {}).get("use_weighted_sampler", False):
        sampler = build_weighted_sampler(train_ds.targets)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_select_loader = DataLoader(
        val_select_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_tune_loader = DataLoader(
        val_tune_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    return {
        "full_ds": full_ds,
        "base_train_ds": base_train_ds,
        "base_val_select_ds": base_val_select_ds,
        "base_val_tune_ds": base_val_tune_ds,
        "base_val_ds": base_val_select_ds,
        "train_ds": train_ds,
        "val_select_ds": val_select_ds,
        "val_tune_ds": val_tune_ds,
        "val_ds": val_select_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_select_loader": val_select_loader,
        "val_tune_loader": val_tune_loader,
        "val_loader": val_select_loader,
        "test_loader": test_loader,
        "classes": classes,
        "class_to_idx": class_to_idx,
    }
