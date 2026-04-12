#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import kagglehub
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, mixed_precision
from tensorflow import keras
import tensorflow as tf
import os
import builtins
import gc
import time
import random
import warnings
import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["ABSL_LOG_LEVEL"] = "3"

SCRIPT_DIR = Path(__file__).resolve(
).parent if "__file__" in globals() else Path.cwd()
LOG_FILE_PATH = SCRIPT_DIR / "EfficientNet_V2.log"
_ORIGINAL_PRINT = builtins.print


def tee_print(*args, **kwargs):
    _ORIGINAL_PRINT(*args, **kwargs)
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(arg) for arg in args) + end
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(message)


with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
    f.write(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

print = tee_print
print(f"Logging to: {LOG_FILE_PATH}")

# KaggleHub: set cache dir BEFORE importing/using kagglehub.
# If you exported KAGGLEHUB_CACHE_DIR in your shell before starting the kernel, it will be used here.
KAGGLEHUB_CACHE_DIR = os.environ.get(
    "KAGGLEHUB_CACHE_DIR", "/mnt/sdd/kaggle_cache")
os.environ["KAGGLEHUB_CACHE_DIR"] = KAGGLEHUB_CACHE_DIR
os.makedirs(KAGGLEHUB_CACHE_DIR, exist_ok=True)
free_gb = shutil.disk_usage(KAGGLEHUB_CACHE_DIR).free / (1024**3)
print(f"KAGGLEHUB_CACHE_DIR={KAGGLEHUB_CACHE_DIR} | free={free_gb:.1f} GB")


tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)
os.environ["TF_AUTOGRAPH_VERBOSITY"] = "0"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {len(gpus)}")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices in strategy: {strategy.num_replicas_in_sync}")

mixed_precision.set_global_policy("mixed_float16")


def save_figure(filename: str, fig=None):
    fig = fig or plt.gcf()
    output_path = SCRIPT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")


KAGGLE_DATASET = "ayushmandatta1/deepdetect-2025"


def resolve_dataset_dirs():
    # output_dir controls where the extracted dataset is placed; cache dir controls where the archive is cached.
    root = Path(kagglehub.dataset_download(
        KAGGLE_DATASET,
        output_dir=KAGGLEHUB_CACHE_DIR,
    ))
    print(f"Dataset downloaded to: {root}")

    candidate_roots = [
        root / "ddata",
        root / "deepdetect-2025" / "ddata",
        root / "deepdetect_2025" / "ddata",
    ]

    data_root = next((p for p in candidate_roots if p.exists()), None)
    if data_root is None:
        matches = list(root.rglob("ddata"))
        if not matches:
            raise FileNotFoundError(
                f"Could not find 'ddata' inside downloaded dataset at {root}"
            )
        data_root = matches[0]

    train_real = data_root / "train" / "real"
    train_fake = data_root / "train" / "fake"
    test_real = data_root / "test" / "real"
    test_fake = data_root / "test" / "fake"

    required_dirs = [train_real, train_fake, test_real, test_fake]
    missing_dirs = [str(p) for p in required_dirs if not p.exists()]
    if missing_dirs:
        raise FileNotFoundError(
            "Missing required dataset directories:\n" + "\n".join(missing_dirs)
        )

    return tuple(str(p) for p in required_dirs)


TRAIN_REAL_DIR, TRAIN_FAKE_DIR, TEST_REAL_DIR, TEST_FAKE_DIR = resolve_dataset_dirs()

SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
EARLY_STOP_PAT = 5
AUTOTUNE = tf.data.AUTOTUNE
CACHE_DIR = Path(".tf_data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("DONE WITH DATA")


# ### Split Dataset

# In[ ]:


VALID_EXT = (".jpg", ".jpeg", ".png")


def get_image_paths(directory):
    return [
        str(p) for p in Path(directory).glob("*")
        if p.suffix.lower() in VALID_EXT
    ]


train_real_paths = get_image_paths(TRAIN_REAL_DIR)
train_fake_paths = get_image_paths(TRAIN_FAKE_DIR)
test_real_paths = get_image_paths(TEST_REAL_DIR)
test_fake_paths = get_image_paths(TEST_FAKE_DIR)

train_paths = train_real_paths + train_fake_paths
train_labels = [0] * len(train_real_paths) + [1] * len(train_fake_paths)

TE_PATHS = test_real_paths + test_fake_paths
TE_LABELS = [0] * len(test_real_paths) + [1] * len(test_fake_paths)

TR_PATHS, VA_PATHS, TR_LABELS, VA_LABELS = train_test_split(
    train_paths,
    train_labels,
    test_size=0.2,
    stratify=train_labels,
    random_state=SEED
)

print(f"Train {len(TR_PATHS)} | Val {len(VA_PATHS)} | Test {len(TE_PATHS)}")


# ### Data Augmentation Pipeline

# In[ ]:


def _load_and_resize(path, label, img_size):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)
    return img, label


def _augment(img, label, img_size):
    scale = tf.random.uniform([], 0.80, 1.0)
    crop_s = tf.cast(tf.cast(img_size, tf.float32) * scale, tf.int32)
    crop_s = tf.maximum(crop_s, 1)
    img = tf.image.random_crop(img, size=[crop_s, crop_s, 3])
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    if hasattr(tf.image, "gaussian_filter2d"):
        sigma = tf.random.uniform([], 0.1, 2.0)
        img = tf.image.gaussian_filter2d(img, filter_shape=3, sigma=sigma)
    return img, label


def make_dataset(paths, labels, img_size: int, augment: bool, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: _load_and_resize(p, l, img_size),
        num_parallel_calls=AUTOTUNE
    ).apply(tf.data.experimental.ignore_errors())
    ds = ds.cache(str(CACHE_DIR / f"cache_{img_size}"))
    if augment:
        ds = ds.map(
            lambda img, lbl: _augment(img, lbl, img_size),
            num_parallel_calls=AUTOTUNE
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


_dataset_cache: dict = {}


def get_datasets(img_size: int):
    if img_size in _dataset_cache:
        print(f"  [cache hit] reusing datasets for img_size={img_size}")
        return _dataset_cache[img_size]

    train_ds = make_dataset(TR_PATHS, TR_LABELS, img_size,
                            augment=True,  shuffle=True)
    val_ds = make_dataset(VA_PATHS, VA_LABELS, img_size,
                          augment=False, shuffle=False)
    test_ds = make_dataset(TE_PATHS, TE_LABELS, img_size,
                           augment=False, shuffle=False)

    _dataset_cache[img_size] = (train_ds, val_ds, test_ds)
    return train_ds, val_ds, test_ds


# ### Model Builder

# In[ ]:


EFFICIENT_NETS = {
    "b0": keras.applications.EfficientNetB0,
    "b2": keras.applications.EfficientNetB2,
    "b3": keras.applications.EfficientNetB3,
}


def build_model(variant="b0", training_type="partial", dropout=0.3, img_size=224):
    cls = EFFICIENT_NETS[variant]

    # Temporarily disable mixed precision to build EfficientNet base
    mixed_precision.set_global_policy("float32")

    base = cls(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling="avg"
    )

    # Restore mixed precision for training
    mixed_precision.set_global_policy("mixed_float16")

    if training_type == "head":
        base.trainable = False

    elif training_type == "partial":
        base.trainable = False
        for layer in base.layers[-30:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    else:
        base.trainable = True
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    x = base.output
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    return keras.Model(inputs=base.input, outputs=output)


# ### Training Helpers

# In[ ]:


def get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adamw":
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    elif name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    elif name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"Unknown optimiser: {name}")


def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PAT,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-7, verbose=1),
    ]


def train_model(model, train_ds, val_ds, extra_callbacks=None):
    cbs = get_callbacks() + (extra_callbacks or [])
    t0 = time.time()
    hist = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=cbs, verbose=1
    )
    avg_t = (time.time() - t0) / len(hist.epoch)
    return hist, avg_t


def evaluate_model(model, test_ds, true_labels):
    results = model.evaluate(test_ds, verbose=0, return_dict=True)
    loss = results["loss"]
    acc = results.get("accuracy", results.get("acc", float("nan")))
    t0 = time.time()
    y_prob = model.predict(test_ds, verbose=0).ravel()
    inf_ms = (time.time() - t0) / len(true_labels) * 1000
    return loss, acc, np.array(true_labels), y_prob, inf_ms


def cleanup(model):
    del model
    keras.backend.clear_session()
    gc.collect()


def plot_loss_curve(histories: dict, title: str, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for label, hist in histories.items():
        h = hist.history if hasattr(hist, "history") else hist
        if "loss" not in h or "val_loss" not in h:
            continue
        axes[0].plot(h["loss"],     label=label, linewidth=2)
        axes[1].plot(h["val_loss"], label=label, linewidth=2)

    for ax, split in zip(axes, ["Train Loss", "Val Loss"]):
        ax.set_title(split)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(filename, fig)
    plt.show()


# ### Experiment 1 - Training Type
# #### Head-Only vs Partial vs Full Finetune

# In[ ]:


print("\n" + "=" * 60)
print("EXPERIMENT 1 — Training Type Comparison")
print("=" * 60)

EXP1_TYPES = ["head", "partial", "full"]
exp1_results = {}

for ttype in EXP1_TYPES:
    print(f"\nTraining Type = {ttype}")
    train_ds, val_ds, test_ds = get_datasets(224)

    with strategy.scope():
        model = build_model("b0", ttype, 0.3, 224)
        model.compile(
            optimizer=get_optimizer("adamw", 5e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    hist, avg_t = train_model(model, train_ds, val_ds)
    _, acc, _, _, _ = evaluate_model(model, test_ds, TE_LABELS)
    exp1_results[ttype] = {"acc": acc, "epoch_time": avg_t}
    print(f"Test acc={acc:.4f}  |  Avg epoch={avg_t:.1f}s")
    cleanup(model)


# ### Experiment 2 - Optimizer Comparison
# #### AdamW vs SGD vs RMSProp

# In[ ]:


print("\n" + "=" * 60)
print("EXPERIMENT 2 — Optimizer Comparison")
print("=" * 60)

EXP2_OPTS = ["adamw", "sgd", "rmsprop"]
exp2_results = {}
exp2_histories = {}

for opt_name in EXP2_OPTS:
    print(f"\nOptimizer = {opt_name}")
    train_ds, val_ds, test_ds = get_datasets(224)

    with strategy.scope():
        model = build_model("b0", "partial", 0.3, 224)
        model.compile(
            optimizer=get_optimizer(opt_name, 5e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    hist, _ = train_model(model, train_ds, val_ds)
    _, acc, _, _, _ = evaluate_model(model, test_ds, TE_LABELS)
    exp2_results[opt_name] = acc
    exp2_histories[opt_name] = hist
    print(f"Test acc={acc:.4f}")
    cleanup(model)

print("\n  ── Experiment 2 Summary ──")
print(f"  {'Optimizer':<12} {'Test Acc':>10}")
for k, v in exp2_results.items():
    print(f"  {k:<12} {v:>10.4f}")

plot_loss_curve(exp2_histories, "Exp 2 — Optimizer: Loss Curves",
                "exp2_optimizer_loss_curves.png")


# ### Experiment 3 - Model Scale Comparison
# #### B0 vs B2 vs B3

# In[ ]:


print("\n" + "=" * 60)
print("EXPERIMENT 3 — Model Scale Comparison")
print("=" * 60)

VARIANT_IMG_SIZE = {"b0": 224, "b2": 260, "b3": 300}
EXP3_VARIANTS = ["b0", "b2", "b3"]
exp3_results = {}

for variant in EXP3_VARIANTS:
    img_size = VARIANT_IMG_SIZE[variant]
    print(f"\nModel Variant = EfficientNet-{variant.upper()}")
    train_ds, val_ds, test_ds = get_datasets(img_size)

    with strategy.scope():
        model = build_model(variant, "partial", 0.3, img_size)
        model.compile(
            optimizer=get_optimizer("adamw", 5e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    hist, avg_t = train_model(model, train_ds, val_ds)
    _, acc, _, _, inf_t = evaluate_model(model, test_ds, TE_LABELS)
    exp3_results[variant] = {"acc": acc, "epoch_time": avg_t, "inf_ms": inf_t}
    print(
        f"Test acc={acc:.4f}  |  Avg epoch={avg_t:.1f}s  |  Inf={inf_t:.3f}ms/img")
    cleanup(model)

print("\n  ── Experiment 3 Summary ──")
print(f"  {'Variant':<8} {'Test Acc':>10} {'Avg Epoch Train Time(s)':>15} {'Inference Time (ms/img)':>14}")
for k, v in exp3_results.items():
    print(
        f"  {k:<8} {v['acc']:>10.4f} {v['epoch_time']:>15.1f} {v['inf_ms']:>14.3f}")


# ### Experiment 4 - Learning Rate Tuning
# #### 1e-4 vs 5e-4 vs 1e-3

# In[ ]:


print("\n" + "=" * 60)
print("EXPERIMENT 4 — Learning Rate Tuning")
print("=" * 60)

EXP4_LRS = [1e-4, 5e-4, 1e-3]
exp4_results = {}
exp4_histories = {}

for lr in EXP4_LRS:
    label = f"{lr:.0e}"
    print(f"\nLearning Rate = {label}")
    train_ds, val_ds, test_ds = get_datasets(260)

    with strategy.scope():
        model = build_model("b2", "partial", 0.3, 260)
        model.compile(
            optimizer=get_optimizer("adamw", lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    hist, _ = train_model(model, train_ds, val_ds)
    _, acc, _, _, _ = evaluate_model(model, test_ds, TE_LABELS)
    exp4_results[label] = acc
    exp4_histories[label] = hist
    print(f"    Test acc={acc:.4f}")
    cleanup(model)

print("\n  ── Experiment 4 Summary ──")
print(f"  {'LR':<8} {'Test Acc':>10}")
for k, v in exp4_results.items():
    print(f"  {k:<8} {v:>10.4f}")

plot_loss_curve(exp4_histories, "Exp 4 — LR Tuning: Loss Curves",
                "exp4_lr_tuning_loss_curves.png")


# ### Dropout Tuning
# #### 0.2 vs 0.3 vs 0.4

# In[ ]:


print("\n" + "=" * 60)
print("EXPERIMENT 5 — Dropout Tuning")
print("=" * 60)

EXP5_DROPOUTS = [0.2, 0.3, 0.4]
exp5_results = {}

for drop in EXP5_DROPOUTS:
    label = f"drop={drop}"
    print(f"\nDropout = {drop}")
    train_ds, val_ds, test_ds = get_datasets(260)

    with strategy.scope():
        model = build_model("b2", "partial", drop, 260)
        model.compile(
            optimizer=get_optimizer("adamw", 5e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    hist, _ = train_model(model, train_ds, val_ds)
    _, acc, _, _, _ = evaluate_model(model, test_ds, TE_LABELS)
    exp5_results[label] = acc
    print(f"    Test acc={acc:.4f}")
    cleanup(model)

print("\n  ── Experiment 5 Summary ──")
print(f"  {'Dropout':<12} {'Test Acc':>10}")
for k, v in exp5_results.items():
    print(f"  {k:<12} {v:>10.4f}")


# ### Final Model Training

# In[ ]:


print("\n" + "=" * 60)
print("FINAL MODEL — B2 | 260px | AdamW | 5e-4 | partial | drop=0.3")
print("=" * 60)

train_ds, val_ds, test_ds = get_datasets(260)

with strategy.scope():
    final_model = build_model("b2", "partial", 0.3, 260)
    final_model.compile(
        optimizer=get_optimizer("adamw", 5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

final_hist, _ = train_model(final_model, train_ds, val_ds)

test_loss, test_acc, y_true, y_prob, _ = evaluate_model(
    final_model, test_ds, TE_LABELS)
y_pred = (y_prob >= 0.5).astype(int)

print(f"\n  Final  Loss={test_loss:.4f}  |  Acc={test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))


# ### Final Training Results

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (tr_key, va_key), title in zip(
    axes,
    [("loss", "val_loss"), ("accuracy", "val_accuracy")],
    ["Loss", "Accuracy"]
):
    ax.plot(final_hist.history[tr_key], label="Train", linewidth=2)
    ax.plot(final_hist.history[va_key], label="Val",
            linewidth=2, linestyle="--")
    ax.set_title(f"Final — {title}")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True)
fig.tight_layout()
save_figure("final_training_loss_accuracy.png", fig)
plt.show()

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Final Model — Confusion Matrix")
fig.tight_layout()
save_figure("final_confusion_matrix.png", fig)
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Final Model — ROC Curve")
ax.legend()
ax.grid(True)
fig.tight_layout()
save_figure("final_roc_curve.png", fig)
plt.show()

precision, recall, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Final Model — Precision-Recall Curve")
ax.legend()
ax.grid(True)
fig.tight_layout()
save_figure("final_precision_recall_curve.png", fig)
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(y_prob[y_true == 0], bins=40, alpha=0.6,
        label="Real", color="steelblue")
ax.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="Fake", color="coral")
ax.axvline(0.5, color="black", linestyle="--",
           linewidth=1.2, label="Threshold = 0.5")
ax.set_xlabel("Predicted P(Fake)")
ax.set_ylabel("Count")
ax.set_title("Final Model — Score Distribution")
ax.legend()
ax.grid(True)
fig.tight_layout()
save_figure("final_score_distribution.png", fig)
plt.show()

print("\n" + "=" * 60)
print("FULL EXPERIMENT SUMMARY")
print("=" * 60)

print(f"\n  Exp 1 — Training Type")
print(f"  {'Type':<10} {'Test Acc':>10} {'Avg Epoch (s)':>15}")
for k, v in exp1_results.items():
    print(f"  {k:<10} {v['acc']:>10.4f} {v['epoch_time']:>15.1f}")

print(f"\n  Exp 2 — Optimizer")
print(f"  {'Optimizer':<12} {'Test Acc':>10}")
for k, v in exp2_results.items():
    print(f"  {k:<12} {v:>10.4f}")

print(f"\n  Exp 3 — Model Scale")
print(f"  {'Variant':<8} {'Test Acc':>10} {'Avg Epoch (s)':>15} {'Inf (ms/img)':>14}")
for k, v in exp3_results.items():
    print(
        f"  {k:<8} {v['acc']:>10.4f} {v['epoch_time']:>15.1f} {v['inf_ms']:>14.3f}")

print(f"\n  Exp 4 — Learning Rate")
print(f"  {'LR':<8} {'Test Acc':>10}")
for k, v in exp4_results.items():
    print(f"  {k:<8} {v:>10.4f}")

print(f"\n  Exp 5 — Dropout")
print(f"  {'Dropout':<12} {'Test Acc':>10}")
for k, v in exp5_results.items():
    print(f"  {k:<12} {v:>10.4f}")

print(
    f"\n  Final Model  |  Test Acc={test_acc:.4f}  AUC={roc_auc:.4f}  AP={ap:.4f}")
print("\nDone ✓")
