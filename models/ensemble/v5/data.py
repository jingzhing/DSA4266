import json
import os
import random
from typing import List

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TransformSubset(Dataset):
    def __init__(self, subset, transform, include_path=False):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = list(subset.indices)
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
        with Image.open(path) as img:
            img = img.convert("RGB")
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
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        if self.include_path:
            return img, target, path
        return img, target


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(img_size, aug_cfg):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=aug_cfg.get("horizontal_flip", 0.5)),
        transforms.ToTensor(),
    ])


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def _stratified_three_way_split(targets, val_select_ratio, val_tune_ratio, seed):
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_select_indices: List[int] = []
    val_tune_indices: List[int] = []

    for cls in np.unique(targets):
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)
        n = len(cls_indices)
        n_select = max(1, int(round(n * val_select_ratio)))
        n_tune = max(1, int(round(n * val_tune_ratio)))
        if n_select + n_tune >= n:
            n_select = max(1, min(n - 2, n_select))
            n_tune = max(1, min(n - n_select - 1, n_tune))
        val_select_cls = cls_indices[:n_select]
        val_tune_cls = cls_indices[n_select:n_select + n_tune]
        train_cls = cls_indices[n_select + n_tune:]
        train_indices.extend(train_cls.tolist())
        val_select_indices.extend(val_select_cls.tolist())
        val_tune_indices.extend(val_tune_cls.tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_select_indices)
    rng.shuffle(val_tune_indices)
    return train_indices, val_select_indices, val_tune_indices


def load_or_create_fixed_split(full_ds, cfg):
    split_path = cfg["split_path"]
    os.makedirs(os.path.dirname(split_path), exist_ok=True)

    current_paths = [full_ds.samples[i][0] for i in range(len(full_ds))]
    current_signature = {
        "dataset_size": len(full_ds),
        "train_dir": str(cfg["train_dir"]),
        "first_20_paths": current_paths[:20],
        "last_20_paths": current_paths[-20:] if len(current_paths) >= 20 else current_paths,
    }

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        saved_signature = payload.get("dataset_signature", {})
        if (
            saved_signature.get("dataset_size") == current_signature["dataset_size"]
            and saved_signature.get("train_dir") == current_signature["train_dir"]
            and saved_signature.get("first_20_paths") == current_signature["first_20_paths"]
            and saved_signature.get("last_20_paths") == current_signature["last_20_paths"]
        ):
            train_indices = payload["train_indices"]
            val_select_indices = payload["val_select_indices"]
            val_tune_indices = payload["val_tune_indices"]
            if len(train_indices) + len(val_select_indices) + len(val_tune_indices) != len(full_ds):
                raise RuntimeError("Saved split does not match current dataset size")
            return train_indices, val_select_indices, val_tune_indices

        print("Existing split file does not match current dataset. Regenerating split.")

    train_indices, val_select_indices, val_tune_indices = _stratified_three_way_split(
        full_ds.targets,
        cfg["val_select_ratio"],
        cfg["val_tune_ratio"],
        cfg["seed"],
    )

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": cfg["seed"],
            "dataset_signature": current_signature,
            "val_select_ratio": cfg["val_select_ratio"],
            "val_tune_ratio": cfg["val_tune_ratio"],
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


def _make_loader(ds, batch_size, shuffle, sampler, num_workers, pin_memory):
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
    )


def build_single_model_dataloaders(cfg):
    model_cfg = cfg["model"]
    aug_cfg = cfg.get("augment", {})

    print("Using train_dir:", cfg["train_dir"])
    print("Using test_dir:", cfg["test_dir"])
    print("Using split_path:", cfg["split_path"])

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

    train_loader = _make_loader(train_ds, cfg["batch_size"], shuffle, sampler, cfg["num_workers"], cfg.get("pin_memory", True))
    val_select_loader = _make_loader(val_select_ds, cfg["batch_size"], False, None, cfg["num_workers"], cfg.get("pin_memory", True))
    val_tune_loader = _make_loader(val_tune_ds, cfg["batch_size"], False, None, cfg["num_workers"], cfg.get("pin_memory", True))
    test_loader = _make_loader(test_ds, cfg["batch_size"], False, None, cfg["num_workers"], cfg.get("pin_memory", True))

    return {
        "full_ds": full_ds,
        "base_train_ds": base_train_ds,
        "base_val_select_ds": base_val_select_ds,
        "base_val_tune_ds": base_val_tune_ds,
        "train_ds": train_ds,
        "val_select_ds": val_select_ds,
        "val_tune_ds": val_tune_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_select_loader": val_select_loader,
        "val_tune_loader": val_tune_loader,
        "test_loader": test_loader,
        "classes": classes,
        "class_to_idx": class_to_idx,
    }
