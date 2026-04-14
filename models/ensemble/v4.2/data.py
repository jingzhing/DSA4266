import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


class LegacyPathDataset(Dataset):
    def __init__(self, paths, labels, transform, include_path=False):
        self.paths = list(paths)
        self.targets = [int(x) for x in labels]
        self.transform = transform
        self.include_path = include_path
        self.classes = ["real", "fake"]
        self.class_to_idx = {"real": 0, "fake": 1}
        self.samples = list(zip(self.paths, self.targets))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        target = self.targets[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        if self.include_path:
            return img, target, path
        return img, target


class LegacyDatasetInfo:
    def __init__(self, train_paths, train_labels, val_paths, val_labels, test_paths, test_labels):
        self.train_paths = list(train_paths)
        self.train_labels = [int(x) for x in train_labels]
        self.val_paths = list(val_paths)
        self.val_labels = [int(x) for x in val_labels]
        self.test_paths = list(test_paths)
        self.test_labels = [int(x) for x in test_labels]
        self.classes = ["real", "fake"]
        self.class_to_idx = {"real": 0, "fake": 1}
        self.targets = self.train_labels + self.val_labels
        self.samples = list(zip(self.train_paths + self.val_paths, self.targets))


class _LegacyTrainSummary:
    def __init__(self, info):
        self.paths = info.train_paths
        self.targets = info.train_labels
        self.classes = info.classes
        self.class_to_idx = info.class_to_idx
        self.samples = list(zip(self.paths, self.targets))

    def __len__(self):
        return len(self.paths)


class _LegacyValSummary:
    def __init__(self, info):
        self.paths = info.val_paths
        self.targets = info.val_labels
        self.classes = info.classes
        self.class_to_idx = info.class_to_idx
        self.samples = list(zip(self.paths, self.targets))

    def __len__(self):
        return len(self.paths)


class _LegacyFullSummary:
    def __init__(self, info):
        self.paths = info.train_paths + info.val_paths
        self.targets = info.train_labels + info.val_labels
        self.classes = info.classes
        self.class_to_idx = info.class_to_idx
        self.samples = list(zip(self.paths, self.targets))

    def __len__(self):
        return len(self.paths)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(img_size, aug_cfg):
    color = aug_cfg.get("color_jitter", 0.0)
    erasing = aug_cfg.get("random_erasing", 0.0)
    legacy_mode = aug_cfg.get("legacy_effnet_mode", False)

    ops = []
    if legacy_mode:
        ops.append(transforms.Resize((img_size, img_size)))
        ops.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
    else:
        ops.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))

    ops.append(transforms.RandomHorizontalFlip(p=aug_cfg.get("horizontal_flip", 0.5)))

    if aug_cfg.get("vertical_flip", 0.0) > 0:
        ops.append(transforms.RandomVerticalFlip(p=aug_cfg["vertical_flip"]))

    if aug_cfg.get("gaussian_blur", {}).get("enabled", False):
        blur_cfg = aug_cfg["gaussian_blur"]
        kernel_size = blur_cfg.get("kernel_size", 3)
        sigma = blur_cfg.get("sigma", (0.1, 2.0))
        ops.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma))

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
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if erasing > 0:
        ops.append(transforms.RandomErasing(p=erasing))

    return transforms.Compose(ops)


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def stratified_split_indices(targets, val_ratio, seed):
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for cls in np.unique(targets):
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)
        n_val_cls = max(1, int(round(len(cls_indices) * val_ratio)))
        val_cls = cls_indices[:n_val_cls]
        train_cls = cls_indices[n_val_cls:]
        train_indices.extend(train_cls.tolist())
        val_indices.extend(val_cls.tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def load_or_create_fixed_split(full_ds, cfg):
    split_path = cfg.get("val_split_path")
    if split_path is None:
        return stratified_split_indices(full_ds.targets, cfg["val_ratio"], cfg["seed"])
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        train_indices = payload["train_indices"]
        val_indices = payload["val_indices"]
        if len(train_indices) + len(val_indices) != len(full_ds):
            raise RuntimeError("Saved validation split does not match current dataset size")
        return train_indices, val_indices
    train_indices, val_indices = stratified_split_indices(full_ds.targets, cfg["val_ratio"], cfg["seed"])
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": cfg["seed"],
            "val_ratio": cfg["val_ratio"],
            "dataset_size": len(full_ds),
            "train_indices": train_indices,
            "val_indices": val_indices,
        }, f, indent=2)
    return train_indices, val_indices


def collect_image_paths(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(str(p) for p in folder.glob("*") if p.suffix.lower() in VALID_EXTS)


def load_or_create_legacy_path_split(cfg):
    train_dir = Path(cfg["train_dir"])
    test_dir = Path(cfg["test_dir"])
    split_path = cfg.get("val_split_path")

    train_real_paths = collect_image_paths(train_dir / "real")
    train_fake_paths = collect_image_paths(train_dir / "fake")
    test_real_paths = collect_image_paths(test_dir / "real")
    test_fake_paths = collect_image_paths(test_dir / "fake")

    if not train_real_paths or not train_fake_paths:
        raise RuntimeError("Expected train_dir to contain non-empty real and fake subfolders")
    if not test_real_paths or not test_fake_paths:
        raise RuntimeError("Expected test_dir to contain non-empty real and fake subfolders")

    train_paths = train_real_paths + train_fake_paths
    train_labels = [0] * len(train_real_paths) + [1] * len(train_fake_paths)
    test_paths = test_real_paths + test_fake_paths
    test_labels = [0] * len(test_real_paths) + [1] * len(test_fake_paths)

    if split_path is None:
        tr_paths, va_paths, tr_labels, va_labels = train_test_split(
            train_paths,
            train_labels,
            test_size=cfg["val_ratio"],
            stratify=train_labels,
            random_state=cfg["seed"],
        )
        return LegacyDatasetInfo(tr_paths, tr_labels, va_paths, va_labels, test_paths, test_labels)

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("n_train_total") != len(train_paths):
            raise RuntimeError("Saved legacy split does not match current train dataset size")
        return LegacyDatasetInfo(
            payload["train_paths"],
            payload["train_labels"],
            payload["val_paths"],
            payload["val_labels"],
            test_paths,
            test_labels,
        )

    tr_paths, va_paths, tr_labels, va_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=cfg["val_ratio"],
        stratify=train_labels,
        random_state=cfg["seed"],
    )
    payload = {
        "seed": cfg["seed"],
        "val_ratio": cfg["val_ratio"],
        "n_train_total": len(train_paths),
        "train_paths": list(tr_paths),
        "train_labels": [int(x) for x in tr_labels],
        "val_paths": list(va_paths),
        "val_labels": [int(x) for x in va_labels],
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return LegacyDatasetInfo(tr_paths, tr_labels, va_paths, va_labels, test_paths, test_labels)


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


def _build_legacy_single_model_dataloaders(cfg):
    model_cfg = cfg["model"]
    aug_cfg = cfg.get("augment", {})
    info = load_or_create_legacy_path_split(cfg)

    train_tf = build_train_transform(model_cfg["img_size"], aug_cfg)
    eval_tf = build_eval_transform(model_cfg["img_size"])

    train_ds = LegacyPathDataset(info.train_paths, info.train_labels, train_tf, include_path=False)
    val_ds = LegacyPathDataset(info.val_paths, info.val_labels, eval_tf, include_path=True)
    test_ds = LegacyPathDataset(info.test_paths, info.test_labels, eval_tf, include_path=True)

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
    val_loader = DataLoader(
        val_ds,
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
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "classes": info.classes,
        "class_to_idx": info.class_to_idx,
        "full_ds": _LegacyFullSummary(info),
        "base_train_ds": _LegacyTrainSummary(info),
        "base_val_ds": _LegacyValSummary(info),
        "data_mode": "legacy_path_lists",
    }


def _build_imagefolder_single_model_dataloaders(cfg):
    model_cfg = cfg["model"]
    aug_cfg = cfg.get("augment", {})

    full_ds = ImageFolder(cfg["train_dir"])
    classes = full_ds.classes
    class_to_idx = full_ds.class_to_idx
    if "fake" not in class_to_idx or "real" not in class_to_idx:
        raise RuntimeError(f"Expected classes fake/real. Got: {classes} mapping={class_to_idx}")

    train_indices, val_indices = load_or_create_fixed_split(full_ds, cfg)
    base_train_ds = Subset(full_ds, train_indices)
    base_train_ds.indices = train_indices
    base_val_ds = Subset(full_ds, val_indices)
    base_val_ds.indices = val_indices

    train_tf = build_train_transform(model_cfg["img_size"], aug_cfg)
    eval_tf = build_eval_transform(model_cfg["img_size"])

    train_ds = TransformSubset(base_train_ds, train_tf, include_path=False)
    val_ds = TransformSubset(base_val_ds, eval_tf, include_path=True)
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
    val_loader = DataLoader(
        val_ds,
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
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "full_ds": full_ds,
        "base_train_ds": base_train_ds,
        "base_val_ds": base_val_ds,
        "data_mode": "imagefolder",
    }


def build_single_model_dataloaders(cfg):
    data_mode = cfg.get("data", {}).get("mode", "imagefolder")
    if data_mode == "legacy_path_lists":
        return _build_legacy_single_model_dataloaders(cfg)
    return _build_imagefolder_single_model_dataloaders(cfg)
