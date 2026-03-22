from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from pipeline.metrics import compute_binary_metrics, find_best_threshold, sigmoid


def _binary_label_from_class_idx(class_idx: np.ndarray, fake_idx: int) -> np.ndarray:
    return (class_idx.astype(int) == int(fake_idx)).astype(int)


def _load_image_paths_for_inference(input_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise RuntimeError(f"No images found in input directory: {input_dir}")
    return image_paths


def train_model(model_name: str, cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    if model_name == "swin":
        return _train_swin(cfg, prepared_root, run_dir)
    if model_name == "efficientnet":
        return _train_efficientnet(cfg, prepared_root, run_dir)
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model(model_name: str, cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    if model_name == "swin":
        return _evaluate_swin(cfg, prepared_root, run_dir)
    if model_name == "efficientnet":
        return _evaluate_efficientnet(cfg, prepared_root, run_dir)
    raise ValueError(f"Unsupported model: {model_name}")


def infer_model(
    model_name: str,
    cfg: Dict[str, Any],
    run_dir: Path,
    input_dir: Path,
) -> List[Dict[str, Any]]:
    if model_name == "swin":
        return _infer_swin(cfg, run_dir, input_dir)
    if model_name == "efficientnet":
        return _infer_efficientnet(cfg, run_dir, input_dir)
    raise ValueError(f"Unsupported model: {model_name}")


def _train_swin(cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    import timm

    model_cfg = cfg["models"]["swin"]
    seed = int(cfg["project"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dir = prepared_root / "train"
    val_dir = prepared_root / "val"
    class_names = cfg["data"]["class_names"]
    fake_idx = class_names.index("fake")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(model_cfg["img_size"], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((model_cfg["img_size"], model_cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds = ImageFolder(str(val_dir), transform=eval_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=model_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=model_cfg["num_workers"],
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_cfg["model_name"], pretrained=True, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_cfg["lr"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_score = -1.0
    best_threshold = float(cfg["training"]["default_threshold"])
    best_auc = float("nan")
    best_epoch = -1
    checkpoint_path = run_dir / "model_checkpoint.pt"

    for epoch in range(int(model_cfg["epochs"])):
        model.train()
        losses: List[float] = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y_bin = (y.numpy() == fake_idx).astype(np.float32)
            y_bin_t = torch.from_numpy(y_bin).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y_bin_t)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_logits: List[np.ndarray] = []
        val_y_raw: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                logits = model(x).squeeze(1).detach().cpu().numpy()
                val_logits.append(logits)
                val_y_raw.append(y.numpy())

        y_raw = np.concatenate(val_y_raw).astype(int)
        y_bin = _binary_label_from_class_idx(y_raw, fake_idx=fake_idx)
        probs = sigmoid(np.concatenate(val_logits))
        threshold, score = find_best_threshold(y_bin, probs)
        metrics = compute_binary_metrics(y_bin, probs, threshold)

        if score > best_val_score:
            best_val_score = score
            best_threshold = threshold
            best_auc = metrics["roc_auc"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg,
                    "class_names": class_names,
                    "fake_idx": fake_idx,
                    "best_threshold": best_threshold,
                    "best_val_balanced_accuracy": best_val_score,
                    "best_val_auc": best_auc,
                },
                checkpoint_path,
            )

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_threshold": float(best_threshold),
        "best_val_balanced_accuracy": float(best_val_score),
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
    }


def _evaluate_swin(cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    import timm

    model_cfg = cfg["models"]["swin"]
    checkpoint_path = run_dir / "model_checkpoint.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Swin checkpoint missing: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    fake_idx = int(ckpt.get("fake_idx", cfg["data"]["class_names"].index("fake")))
    best_threshold = float(ckpt.get("best_threshold", cfg["training"]["default_threshold"]))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_tf = transforms.Compose(
        [
            transforms.Resize((model_cfg["img_size"], model_cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_ds = ImageFolder(str(prepared_root / "test"), transform=test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=model_cfg["num_workers"],
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_cfg["model_name"], pretrained=False, num_classes=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_logits: List[np.ndarray] = []
    all_y_raw: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1).detach().cpu().numpy()
            all_logits.append(logits)
            all_y_raw.append(y.numpy())

    y_raw = np.concatenate(all_y_raw).astype(int)
    y_bin = _binary_label_from_class_idx(y_raw, fake_idx=fake_idx)
    probs = sigmoid(np.concatenate(all_logits))
    metrics = compute_binary_metrics(y_bin, probs, best_threshold)

    samples = test_ds.samples
    predictions: List[Dict[str, Any]] = []
    for i, (path, class_idx) in enumerate(samples):
        prob = float(probs[i])
        pred = int(prob >= best_threshold)
        predictions.append(
            {
                "path": str(path),
                "true_label": int(class_idx == fake_idx),
                "prob_fake": prob,
                "pred_label": pred,
                "threshold": float(best_threshold),
            }
        )
    return {"metrics": metrics, "predictions": predictions}


class _InferenceDataset:
    def __init__(self, image_paths: Sequence[Path], transform: Any):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        import torch

        image = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(image)
        return tensor, str(self.image_paths[idx])


def _infer_swin(cfg: Dict[str, Any], run_dir: Path, input_dir: Path) -> List[Dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import timm

    model_cfg = cfg["models"]["swin"]
    checkpoint_path = run_dir / "model_checkpoint.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Swin checkpoint missing: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    threshold = float(ckpt.get("best_threshold", cfg["training"]["default_threshold"]))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize((model_cfg["img_size"], model_cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    image_paths = _load_image_paths_for_inference(input_dir)
    ds = _InferenceDataset(image_paths, transform)
    loader = DataLoader(ds, batch_size=model_cfg["batch_size"], shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_cfg["model_name"], pretrained=False, num_classes=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1).detach().cpu().numpy()
            probs = sigmoid(logits)
            for path, prob in zip(paths, probs):
                pred = int(prob >= threshold)
                rows.append(
                    {
                        "path": str(path),
                        "prob_fake": float(prob),
                        "pred_label": pred,
                        "threshold": float(threshold),
                    }
                )
    return rows


def _tf_variant(variant: str):
    import tensorflow as tf

    variant = variant.upper()
    if variant == "B0":
        return tf.keras.applications.EfficientNetB0
    if variant == "B1":
        return tf.keras.applications.EfficientNetB1
    if variant == "B2":
        return tf.keras.applications.EfficientNetB2
    if variant == "B3":
        return tf.keras.applications.EfficientNetB3
    raise ValueError(f"Unsupported EfficientNet variant: {variant}")


def _build_efficientnet_model(cfg: Dict[str, Any]):
    import tensorflow as tf

    model_cfg = cfg["models"]["efficientnet"]
    backbone_cls = _tf_variant(model_cfg["variant"])
    inputs = tf.keras.Input(shape=(model_cfg["img_size"], model_cfg["img_size"], 3))
    backbone = backbone_cls(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = not bool(model_cfg["freeze_backbone"])
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(float(model_cfg["dropout"]))(x)
    outputs = tf.keras.layers.Dense(1, dtype="float32")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(model_cfg["lr"])),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _tf_dataset_from_directory(
    root: Path,
    cfg: Dict[str, Any],
    training: bool,
):
    import tensorflow as tf

    model_cfg = cfg["models"]["efficientnet"]
    class_names = cfg["data"]["class_names"]
    ds = tf.keras.utils.image_dataset_from_directory(
        str(root),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(model_cfg["img_size"], model_cfg["img_size"]),
        batch_size=model_cfg["batch_size"],
        shuffle=training,
        seed=int(cfg["project"]["seed"]),
    )
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def _train_efficientnet(cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    model_cfg = cfg["models"]["efficientnet"]
    fake_idx = cfg["data"]["class_names"].index("fake")
    if fake_idx != 1:
        raise RuntimeError("Config class_names must map fake to index 1 for binary label semantics.")

    train_ds = _tf_dataset_from_directory(prepared_root / "train", cfg, training=True)
    val_ds = _tf_dataset_from_directory(prepared_root / "val", cfg, training=False)
    model = _build_efficientnet_model(cfg)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(model_cfg["epochs"]),
        verbose=1,
    )

    logits = model.predict(val_ds, verbose=0).reshape(-1)
    labels = np.concatenate([y.numpy() for _, y in val_ds]).astype(int)
    probs = sigmoid(logits)
    threshold, score = find_best_threshold(labels, probs)
    metrics = compute_binary_metrics(labels, probs, threshold)

    checkpoint_path = run_dir / "model_checkpoint.keras"
    model.save(checkpoint_path)
    metadata_path = run_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "best_threshold": float(threshold),
                "best_val_balanced_accuracy": float(score),
                "best_val_auc": float(metrics["roc_auc"]),
                "history_keys": list(history.history.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "best_threshold": float(threshold),
        "best_val_balanced_accuracy": float(score),
        "best_val_auc": float(metrics["roc_auc"]),
    }


def _evaluate_efficientnet(cfg: Dict[str, Any], prepared_root: Path, run_dir: Path) -> Dict[str, Any]:
    import tensorflow as tf

    checkpoint_path = run_dir / "model_checkpoint.keras"
    metadata_path = run_dir / "model_metadata.json"
    if not checkpoint_path.exists():
        raise RuntimeError(f"EfficientNet checkpoint missing: {checkpoint_path}")
    if not metadata_path.exists():
        raise RuntimeError(f"EfficientNet metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata.get("best_threshold", cfg["training"]["default_threshold"]))

    test_ds = _tf_dataset_from_directory(prepared_root / "test", cfg, training=False)
    model = tf.keras.models.load_model(checkpoint_path)
    logits = model.predict(test_ds, verbose=0).reshape(-1)
    labels = np.concatenate([y.numpy() for _, y in test_ds]).astype(int)
    probs = sigmoid(logits)
    metrics = compute_binary_metrics(labels, probs, threshold)

    # Rebuild path order from directory listing because tf dataset does not retain paths in output tensors.
    class_names = cfg["data"]["class_names"]
    image_paths: List[Path] = []
    true_labels: List[int] = []
    for class_name in class_names:
        class_dir = prepared_root / "test" / class_name
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file():
                image_paths.append(image_path)
                true_labels.append(1 if class_name == "fake" else 0)
    count = min(len(image_paths), len(probs))
    predictions = []
    for i in range(count):
        prob = float(probs[i])
        pred = int(prob >= threshold)
        predictions.append(
            {
                "path": str(image_paths[i]),
                "true_label": int(true_labels[i]),
                "prob_fake": prob,
                "pred_label": pred,
                "threshold": float(threshold),
            }
        )
    return {"metrics": metrics, "predictions": predictions}


def _infer_efficientnet(cfg: Dict[str, Any], run_dir: Path, input_dir: Path) -> List[Dict[str, Any]]:
    import tensorflow as tf

    checkpoint_path = run_dir / "model_checkpoint.keras"
    metadata_path = run_dir / "model_metadata.json"
    if not checkpoint_path.exists():
        raise RuntimeError(f"EfficientNet checkpoint missing: {checkpoint_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata.get("best_threshold", cfg["training"]["default_threshold"]))

    image_paths = _load_image_paths_for_inference(input_dir)
    img_size = int(cfg["models"]["efficientnet"]["img_size"])
    model = tf.keras.models.load_model(checkpoint_path)

    rows: List[Dict[str, Any]] = []
    batch: List[np.ndarray] = []
    batch_paths: List[Path] = []
    batch_size = int(cfg["models"]["efficientnet"]["batch_size"])

    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((img_size, img_size))
        batch.append(np.asarray(img, dtype=np.float32) / 255.0)
        batch_paths.append(path)
        if len(batch) >= batch_size:
            logits = model.predict(np.stack(batch), verbose=0).reshape(-1)
            probs = sigmoid(logits)
            for local_path, prob in zip(batch_paths, probs):
                rows.append(
                    {
                        "path": str(local_path),
                        "prob_fake": float(prob),
                        "pred_label": int(prob >= threshold),
                        "threshold": float(threshold),
                    }
                )
            batch = []
            batch_paths = []

    if batch:
        logits = model.predict(np.stack(batch), verbose=0).reshape(-1)
        probs = sigmoid(logits)
        for local_path, prob in zip(batch_paths, probs):
            rows.append(
                {
                    "path": str(local_path),
                    "prob_fake": float(prob),
                    "pred_label": int(prob >= threshold),
                    "threshold": float(threshold),
                }
            )
    return rows
