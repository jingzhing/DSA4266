# Production-grade AGENTS.md for DeepDetect-2025 Stage-One Pipeline

## Executive summary

This deliverable is a **production-grade, step-by-step** `AGENTS.md` that Codex in VS Code can follow to implement your **Stage One pipeline** (Data layer → Preprocessing layer → Feature variants layer) for the DeepDetect-2025 Kaggle dataset. It is designed to prevent the most common failure modes in synthetic-image detection projects: **data leakage (duplicates, split contamination), shortcut learning (file-type / resolution / JPEG pipeline artifacts correlated with labels), and content bias (semantic distribution mismatch between real vs fake)**—all of which can inflate “high accuracy” without producing a detector that generalizes. These risks are directly aligned with what recent detector research emphasizes: (i) sensitivity to preprocessing/augmentation and the need for careful data handling citeturn2search0, (ii) robustness failure under laundering / impairments and the need to explicitly evaluate and control for it citeturn2search9, and (iii) the core challenge of generalization to **previously unseen generators**, which is primarily constrained by training data diversity and evaluation protocol citeturn2search6.

Key implementation choices are grounded in primary/official documentation:

- **Kaggle dataset download + file listing** uses official Kaggle CLI commands (`kaggle datasets download`, `kaggle datasets files`) citeturn13view0turn9view0.  
- **Reproducible PyTorch dataloading** uses PyTorch’s official reproducibility guidance for `worker_init_fn` and `generator` citeturn0search3.  
- **Preprocessing sizes + normalization per backbone** use TorchVision model weight docs and `weights.transforms()` citeturn5search1turn4search3turn5search2turn5search0turn14search1.  
- **JPEG quantization table access** uses Pillow’s official documentation (`im.quantization`) citeturn2search3.  
- **Perceptual hashing** uses ImageHash’s official docs/pypi for pHash/aHash/dHash/wHash citeturn1search1turn1search12.  
- **CLIP embeddings** use OpenCLIP’s documented `create_model_and_transforms()` citeturn1search2 (CLIP’s original paper for representation motivation citeturn14search0).  
- **Embedding NN search** uses FAISS (official wiki notes `IndexFlatIP` supports cosine when vectors are normalized) citeturn1search20.  
- **Group-aware stratified splitting** uses scikit-learn’s `StratifiedGroupKFold` citeturn3search3turn3search11.  
- **OpenCV channel-order caveat** (BGR on decode) is documented by OpenCV citeturn4search0.

## AGENTS.md

```markdown
# AGENTS.md — Stage One Pipeline (DeepDetect-2025)
**Scope:** Implement Stage One of the project pipeline for DeepDetect-2025:
1) Data layer (ingestion, verification, audit, duplicates, bias checks, splits)
2) Preprocessing layer (deterministic transforms per backbone)
3) Feature variants layer (RGB, residual, FFT magnitude, block-DCT)

**Non-goals (Stage One):** model training, robustness augmentation, fine-tuning, evaluation metrics.

---

## Operating rules (read first)
- Do **NOT** start training until all Stage One artifacts and verification gates pass.
- Treat the dataset as untrusted until:
  - checksums + file manifest are created,
  - format/metadata audit is complete,
  - duplicates are detected and grouped,
  - content-bias diagnostics are run,
  - splits are built with group constraints and verified.

- Keep the test set immutable once created.
- All scripts must be **idempotent**: safe to re-run without corrupting outputs.
- Do not commit raw data to git.

---

## Repository inventory (assume partial progress exists)

### Step 0 — inspect repo structure and existing artifacts
Run:

```bash
git status
git rev-parse --show-toplevel
git ls-files | head -n 50
find . -maxdepth 3 -type d -name "data" -o -name "datasets" -o -name "scripts" -o -name "src" -o -name "configs"
find . -maxdepth 4 -type f -iname "*manifest*" -o -iname "*audit*" -o -iname "*split*" -o -iname "*phash*" -o -iname "*clip*" | sed -n '1,200p'
du -sh data 2>/dev/null || true
```

**If any of these already exist, DO NOT overwrite them silently.**
- If `data/processed/deepdetect_2025/*` exists, log what is present and compare dataset fingerprints before deciding to regenerate.

### Step 0b — required integration edits (create if missing)
Create / confirm these repo paths:
- `scripts/stage1/` (all stage-one scripts)
- `src/dataset/` (python modules used by scripts)
- `data/raw/` (downloaded Kaggle dataset lives here)
- `data/processed/deepdetect_2025/` (generated manifests, audits, embeddings, splits)
- `.gitignore` must include:
  - `data/raw/`
  - `data/processed/`
  - `*.npz`, `*.parquet` if you don’t want them committed
  - `.venv/`, `__pycache__/`

---

## Environment setup (repeatable + minimal)
### Python + deps
Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -V
pip install -U pip wheel
```

Create `requirements_stage1.txt` (if not present) with:

```txt
numpy
pandas
pyarrow
tqdm
pillow
imagehash
scikit-learn
torch
torchvision
open-clip-torch
faiss-cpu
matplotlib
```

Install:

```bash
pip install -r requirements_stage1.txt
python -c "import torch; import torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__)"
```

**Optional speed upgrade (do only after everything works):**
- Install Pillow-SIMD and remove Pillow (advanced; may vary by platform).
- Track it in a separate requirements file to avoid breaking reproducibility.

---

## Stage One — Data layer

### Canonical paths and dataset naming
Use these constants everywhere:

- Dataset handle (Kaggle): `ayushmandatta1/deepdetect-2025`
- Dataset slug: `deepdetect_2025`
- Raw root: `data/raw/deepdetect_2025/`
- Processed root: `data/processed/deepdetect_2025/`

All stage-one artifacts will be derived from a single manifest:
- `data/processed/deepdetect_2025/manifest_v1.json`

---

### Step 1 — Kaggle download (raw ingestion)
#### 1.1 Kaggle auth sanity
Confirm Kaggle CLI works:

```bash
kaggle --help | head
```

If you get 403/Unauthorized, fix credentials first.

#### 1.2 Download + unzip (idempotent)
Create directory and download:

```bash
mkdir -p data/raw/deepdetect_2025
kaggle datasets download -d ayushmandatta1/deepdetect-2025 -p data/raw/deepdetect_2025 --unzip -o
```

Also capture a Kaggle file listing for provenance:

```bash
kaggle datasets files ayushmandatta1/deepdetect-2025 --page-size=200 > data/processed/deepdetect_2025/kaggle_files_page1.txt || true
```

**If dataset has more than 200 files**: implement pagination later. (For most image datasets, folder contents are inside the single zip.)

#### 1.3 Raw file count capture
```bash
find data/raw/deepdetect_2025 -type f | wc -l > data/processed/deepdetect_2025/raw_file_count.txt
find data/raw/deepdetect_2025 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l > data/processed/deepdetect_2025/raw_image_file_count.txt
```

---

### Step 2 — Build dataset manifest (checksums + folder mapping)
**Goal:** Create one JSON manifest containing:
- relative path, absolute path
- file size
- SHA256 checksum
- inferred label (`real`/`fake`)
- inferred source/generator hint (from folder)
- stable sample_id

#### 2.0 Create script: `scripts/stage1/02_build_manifest.py`
Pseudocode:

```python
# inputs: root_dir, output_json, seed
# walk all files under root_dir
# filter by image extensions
# compute sha256 per file (streaming)
# infer label from path tokens (real/fake)
# infer source_hint from folder (customizable rules)
# assign sample_id = stable hash(relpath + sha256) or sequential after sorting
# write manifest_v1.json (sorted by relpath for stability)
# also write manifest_fingerprint.txt = sha256(manifest file bytes)
```

**Implementation detail rules**
- Label inference MUST be a function you can lock down after inspection:
  - start heuristic: if any path part equals `real` -> label=real; else if equals `fake` -> label=fake
  - else label=null and exclude until mapped
- Source hint:
  - start as parent folder name, but you must verify with a human gate.

#### 2.1 Run manifest build
```bash
mkdir -p data/processed/deepdetect_2025
python scripts/stage1/02_build_manifest.py \
  --root data/raw/deepdetect_2025 \
  --out data/processed/deepdetect_2025/manifest_v1.json \
  --compute-sha256
```

#### 2.2 Human verification gate A (folder mapping)
OPEN `manifest_v1.json` and confirm:
- labels are correctly assigned (real/fake)
- no obvious mislabeled folders (e.g., "train/fake" vs "test/fake")

**STOP if:**
- >0.5% images have label=null
- label counts are wildly imbalanced but expected balanced
- folder structure suggests a different labeling scheme

Write decision:
- `data/processed/deepdetect_2025/verification_gate_A_folder_mapping.md`

---

### Step 3 — Programmatic dataset audit (format + metadata + JPEG qtables)
**Goal:** Detect shortcut leakage and pipeline artifacts.

Compute per-image:
- ext, bytes
- width, height, aspect_ratio
- color mode (RGB, RGBA, L, CMYK, etc)
- JPEG quantization tables signature (if JPEG), using Pillow `im.quantization`

#### 3.0 Create script: `scripts/stage1/03_audit_images.py`
Pseudocode:

```python
# load manifest
# for each image:
#   open with PIL.Image.open
#   record width/height/mode/format
#   if JPEG: read im.quantization (tables in zigzag order) and hash them to qsig
#   store status=ok or status=decode_failed
# write audit.parquet
# write audit_summary.json with distributions and label-conditional distributions
```

#### 3.1 Run audit
```bash
python scripts/stage1/03_audit_images.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --out-parquet data/processed/deepdetect_2025/audit_v1.parquet \
  --out-summary data/processed/deepdetect_2025/audit_summary_v1.json
```

#### 3.2 Automated assertions (fail fast)
Create `scripts/stage1/03b_assert_audit.py`:

Assertions:
- decode_failed_rate <= 0.005
- ext distribution by label: Jensen-Shannon divergence <= threshold (start 0.15) → warn if higher
- resolution buckets by label: warn if large separation
- if JPEG qsig exists: warn if top qsig dominates one label (>90%) and is rare in other (<10%)

Run:
```bash
python scripts/stage1/03b_assert_audit.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --audit data/processed/deepdetect_2025/audit_v1.parquet
```

#### 3.3 Human verification gate B (audit summary)
Review `audit_summary_v1.json`:
- confirm file types don’t trivially separate labels (e.g., fake=PNG, real=JPG)
- confirm resolution distributions overlap substantially
- confirm color modes overlap (watch for alpha channel leakage)

Write decision:
- `data/processed/deepdetect_2025/verification_gate_B_audit.md`

---

### Step 4 — Duplicate detection (pHash groups + embedding-based near duplicates)
**Goal:** Build `duplicate_group_id` so duplicates never cross splits.

#### 4.1 Perceptual hash (fast)
Create `scripts/stage1/04_dupes_phash.py`:
- compute pHash (default), also optionally dHash
- create exact-hash groups
- optionally compute near-duplicate edges using Hamming distance threshold (tune later)

Run:
```bash
python scripts/stage1/04_dupes_phash.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --out-parquet data/processed/deepdetect_2025/phash_v1.parquet \
  --hash phash --hash-size 16
```

Output expected:
- `phash_v1.parquet` with columns: sample_id, phash
- `phash_groups_v1.json` mapping hash -> list(sample_id)

#### 4.2 CLIP embeddings (for duplicates + content bias)
Create `scripts/stage1/05_embed_openclip.py`
- use OpenCLIP `create_model_and_transforms`
- batch encode images, normalize embeddings
- save as:
  - `clip_embeds_v1.npz` (X float32, keep_idx sample_id list)
  - OR `clip_embeds_v1.memmap` for large scale (optional)

Run:
```bash
python scripts/stage1/05_embed_openclip.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --out data/processed/deepdetect_2025/clip_embeds_v1.npz \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --batch-size 256 \
  --device cuda
```

#### 4.3 Embedding-based near duplicates (FAISS)
Create `scripts/stage1/06_dupes_faiss.py`:
Algorithm:
- load embeddings X (L2 normalized)
- build `faiss.IndexFlatIP(d)` (cosine via dot product)
- search k nearest for each item (k=5 or 10)
- flag pairs with sim >= threshold (start 0.995; tune by spot-check)
- union-find to build duplicate groups
- output:
  - `embed_dupe_pairs_v1.parquet` (i, j, sim)
  - `duplicate_groups_v1.json` (group_id -> members)
  - `sample_to_dupe_group_v1.json`

Run:
```bash
python scripts/stage1/06_dupes_faiss.py \
  --clip data/processed/deepdetect_2025/clip_embeds_v1.npz \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --out-dir data/processed/deepdetect_2025/dupes_v1 \
  --k 10 \
  --sim-thresh 0.995
```

#### 4.4 Human verification gate C (duplicate thresholds)
Manually inspect ~50 flagged pairs (create a script to dump side-by-side thumbnails).
Confirm threshold isn’t over-grouping unrelated images.

Write decision:
- `data/processed/deepdetect_2025/verification_gate_C_duplicates.md`

---

### Step 5 — Content-bias checks (CLIP + ImageNet embeddings, clustering)
**Goal:** detect semantic/domain shifts that might let models “cheat”.

#### 5.1 CLIP clustering
Create `scripts/stage1/07_bias_cluster_clip.py`:
- load CLIP embeddings
- run k-means (start K=50; break down by label)
- compute per-cluster label purity and entropy
- output:
  - `clip_clusters_v1.parquet` (sample_id, cluster_id)
  - `clip_cluster_stats_v1.json` (purity histograms)
Optional:
- UMAP 2D plot for visualization

Run:
```bash
python scripts/stage1/07_bias_cluster_clip.py \
  --clip data/processed/deepdetect_2025/clip_embeds_v1.npz \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --out-dir data/processed/deepdetect_2025/bias_v1 \
  --k 50 \
  --seed 1337
```

#### 5.2 ImageNet embedding clustering (optional but recommended)
Create `scripts/stage1/08_bias_embed_imagenet.py`:
- use torchvision resnet50 weights
- remove final classifier and take penultimate pooled features (2048-d)
- normalize and save embeddings
Then cluster same way as CLIP.

Run:
```bash
python scripts/stage1/08_bias_embed_imagenet.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --out data/processed/deepdetect_2025/imagenet_embeds_v1.npz \
  --batch-size 256 \
  --device cuda

python scripts/stage1/09_bias_cluster_imagenet.py \
  --emb data/processed/deepdetect_2025/imagenet_embeds_v1.npz \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --out-dir data/processed/deepdetect_2025/bias_v1 \
  --k 50 \
  --seed 1337
```

#### 5.3 Human verification gate D (bias risk decision)
If many clusters have >95% purity by label, you likely have content bias or pipeline bias.
Decide:
- proceed but require cluster-holdout evaluation later, OR
- adjust dataset curation if leakage is severe.

Write:
- `data/processed/deepdetect_2025/verification_gate_D_bias.md`

---

## Stage One — Split builder (reproducible + leakage-safe)

### Split ratios + seed (default)
- seed: `1337`
- ratios:
  - train: 0.80
  - val:   0.10
  - test:  0.10

### Split precedence rules (critical)
Define `group_id` for splitting:
1) if embedding-duplicate groups exist: use `duplicate_group_id`
2) else fallback to pHash groups
3) else each sample is its own group
Optionally:
- if source/generator metadata exists: combine as `group_id = (source_id, duplicate_group_id)` or just `source_id` for strict source-disjoint splits.

### Step 6 — Source metadata detection
Create `scripts/stage1/10_detect_source_metadata.py`:
- inspect manifest paths for generator/source folder patterns
- output candidate `source_id` rules and a summary table

Human gate:
- approve a stable `source_id` extraction rule in config file:
  - `configs/deepdetect_2025_source_rules.yaml`

---

### Step 7A — Group-aware stratified split (default)
**Algorithm:** use StratifiedGroupKFold in two stages to approximate 80/10/10.

Implementation:
- stage1: SGKF with n_splits=10 → choose fold `f_test` as test
- stage2: SGKF with n_splits=9 on remaining → choose fold `f_val` as val
- rest → train

Create `scripts/stage1/11_split_stratified_group.py`:

Pseudocode:

```python
seed = 1337
n_test_folds = 10
# groups = group_id
# y = label (0/1)

sgkf_test = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
folds = list(sgkf_test.split(X=np.zeros(n), y=y, groups=groups))
trainval_idx, test_idx = folds[0]  # choose fold 0 deterministically

# now split trainval into train/val
y_tv = y[trainval_idx]; g_tv = groups[trainval_idx]
sgkf_val = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=seed)
folds2 = list(sgkf_val.split(X=np.zeros(len(trainval_idx)), y=y_tv, groups=g_tv))
train_idx2, val_idx2 = folds2[0]
train_idx = trainval_idx[train_idx2]
val_idx   = trainval_idx[val_idx2]

# write JSON: {split: [sample_id,...]}
```

Run:
```bash
python scripts/stage1/11_split_stratified_group.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --dupes data/processed/deepdetect_2025/dupes_v1/sample_to_dupe_group_v1.json \
  --out data/processed/deepdetect_2025/splits/sgkf_seed1337_v1.json \
  --seed 1337
```

Post-split assertions:
- class ratio within ±1% between splits
- no group_id overlaps across splits
- all sample_ids accounted for exactly once

---

### Step 7B — Source-/generator-disjoint split (if source_id exists)
**Algorithm:** group by `source_id` only, optionally stratify at source-level.

Create `scripts/stage1/12_split_source_disjoint.py`:

Rules:
- no source_id overlaps across splits
- attempt per-split class balance by greedy assignment at source level:
  - compute (real_count, fake_count) per source
  - assign sources to splits to match target totals

Run:
```bash
python scripts/stage1/12_split_source_disjoint.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --source-rules configs/deepdetect_2025_source_rules.yaml \
  --out data/processed/deepdetect_2025/splits/source_disjoint_seed1337_v1.json \
  --seed 1337
```

---

### Step 7C — Cluster-holdout fallback (if no source_id)
**Algorithm (class-conditional):**
- cluster real embeddings into Kr clusters, fake into Kf clusters separately
- sample ~10% clusters per class as test clusters, ~10% as val clusters
- remaining clusters form train
- this forces a semantic/domain shift proxy while preserving label balance

Create:
- `scripts/stage1/13_split_cluster_holdout.py`

Run:
```bash
python scripts/stage1/13_split_cluster_holdout.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --clip-clusters data/processed/deepdetect_2025/bias_v1/clip_clusters_v1.parquet \
  --out data/processed/deepdetect_2025/splits/cluster_holdout_seed1337_v1.json \
  --seed 1337
```

---

### Human verification gate E (final split choice)
Choose ONE split file to be “canonical”:
- `splits/canonical.json` → symlink or copy to chosen split strategy
Explain why in:
- `data/processed/deepdetect_2025/verification_gate_E_split_choice.md`

---

## Stage One — Deterministic dataloaders

### Step 8 — Create dataset class + deterministic DataLoader utilities
Create module: `src/dataset/deepdetect.py`
- loads manifest + chosen split list
- reads images using selected backend (PIL default)
- returns `(image, label, metadata)`

Create module: `src/dataset/repro.py`
- `seed_everything(seed)`
- `seed_worker(worker_id)`
- DataLoader creation with `generator=g`, `worker_init_fn=seed_worker`

Pseudocode:

```python
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws); random.seed(ws)

g = torch.Generator()
g.manual_seed(seed)

dl = DataLoader(ds, shuffle=True, num_workers=..., worker_init_fn=seed_worker, generator=g, persistent_workers=True)
```

### Caching strategy (Stage One)
- Cache heavy *analysis artifacts*:
  - audit parquet, embeddings npz, cluster outputs, split json
- For training-time caching (future stage):
  - prefer shard-based caches (avoid 100k tiny files)
  - formats:
    - `.npz` for embeddings (already)
    - `.parquet` for tabular metadata
    - optional `zarr` for chunked large arrays

---

## Stage One — Preprocessing layer

### Canonical transform policy
- Use TorchVision pretrained weights transforms whenever possible:
  - `weights = ..._Weights.DEFAULT`
  - `preprocess = weights.transforms()`

### Recommended sizes / normalization (documented by torchvision weights)
| Backbone | resize_size | crop_size | mean | std |
|---|---:|---:|---|---|
| EfficientNet-B0 | 256 | 224 | [0.485,0.456,0.406] | [0.229,0.224,0.225] |
| EfficientNet-B3 | 320 | 300 | [0.485,0.456,0.406] | [0.229,0.224,0.225] |
| Swin-T | 232 | 224 | [0.485,0.456,0.406] | [0.229,0.224,0.225] |
| Swin-S | 246 | 224 | [0.485,0.456,0.406] | [0.229,0.224,0.225] |

### Step 9 — Implement preprocess factory
Create `src/dataset/preprocess.py`:

```python
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B3_Weights,
    Swin_T_Weights, Swin_S_Weights
)

def preprocess_for(backbone: str):
    if backbone == "efficientnet_b0":
        return EfficientNet_B0_Weights.DEFAULT.transforms()
    if backbone == "efficientnet_b3":
        return EfficientNet_B3_Weights.DEFAULT.transforms()
    if backbone == "swin_t":
        return Swin_T_Weights.DEFAULT.transforms()
    if backbone == "swin_s":
        return Swin_S_Weights.DEFAULT.transforms()
    raise ValueError(backbone)
```

---

## Stage One — Feature variants layer

### Variants to implement
- RGB tensor (baseline): `x_rgb`
- Residual tensor:
  - median blur subtraction (`k=3` or `k=5`)
  - optional Laplacian high-pass
- FFT magnitude:
  - optional windowing (Hann)
  - fftshift
  - log1p magnitude
  - normalization
- Block DCT (JPEG-aligned):
  - 8×8 blocks on luminance (Y channel)
  - DCT-II with precomputed basis
  - keep low-freq coefficients (e.g., 4×4)

### Step 10 — Create feature functions
Create `src/dataset/features.py` with functions:

#### 10.1 RGB
```python
def feat_rgb(pil_img, preprocess):
    return preprocess(pil_img.convert("RGB"))  # CxHxW float normalized
```

#### 10.2 Residual (median blur subtraction)
Pseudocode:
```python
def feat_residual(rgb_tensor, k=3):
    # rgb_tensor: float, normalized or [0,1] depending on where you compute residual
    # recommended: compute residual BEFORE final normalization
    # use torch median pooling approximation or implement in numpy/cv2
    blur = median_blur(rgb_tensor, k=k)
    res = rgb_tensor - blur
    res = clamp(res, -0.5, 0.5) / 0.5  # scale to [-1,1]
    return res
```

Implementation recommendation:
- simplest: convert to uint8 numpy, apply `cv2.medianBlur`, convert back.

#### 10.3 FFT magnitude (from residual luminance)
Pseudocode:
```python
def feat_fft_mag(residual_tensor):
    # convert residual RGB -> luma
    y = 0.299*r + 0.587*g + 0.114*b
    y = y * hann2d(H,W)
    F = torch.fft.fft2(y)
    M = torch.log1p(torch.abs(F))
    M = torch.fft.fftshift(M)
    M = (M - M.mean()) / (M.std() + 1e-6)
    return M.unsqueeze(0)  # 1xHxW
```

#### 10.4 Block DCT features (8×8)
Vectorized DCT-II via matrix multiply:

Pseudocode:
```python
def dct_matrix(n=8):
    # create orthonormal DCT-II basis T (n x n)
    # T[u,x] = alpha(u) * cos(pi*(2x+1)*u/(2n))

def block_dct(y):
    y = pad_to_multiple(y, 8)
    blocks = view_as_blocks(y, 8)          # (Bh, Bw, 8, 8)
    C = T @ blocks @ T.T                   # broadcast matmul
    C_low = C[..., :4, :4]                 # keep low freqs
    feat = flatten_last2(C_low)            # (Bh, Bw, 16)
    return feat
```

### Cost estimates (theory-level)
| Variant | Time complexity (per image) | Space (per image output) | Notes |
|---|---|---|---|
| RGB preprocess | O(HW) | 3×H×W float | dominated by decode+resize |
| Residual (median k) | ~O(HW·k²) | 3×H×W float | median has higher constant |
| FFT magnitude | O(HW log(HW)) | 1×H×W float | cache recommended if expensive |
| Block DCT (8×8) | ~O(HW) | (H/8×W/8×C) | keep 4×4 low-freq → 16 coeffs/block |

### Caching recommendations for feature variants
- RGB/residual: compute on the fly
- FFT/DCT: optionally cache
  - if caching, avoid per-image tiny files; prefer sharded cache:
    - `data/processed/deepdetect_2025/feature_cache/{variant}/{backbone}/shard_00000.npz`
  - include `manifest_fingerprint` in cache metadata to prevent stale reuse

### Step 11 — Micro-benchmark script (MANDATORY)
Create `scripts/stage1/20_benchmark_features.py`:
- sample N=256 images
- measure per-variant median latency on CPU and GPU
- output JSON: `feature_bench_v1.json`

Run:
```bash
python scripts/stage1/20_benchmark_features.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --n 256 \
  --out data/processed/deepdetect_2025/feature_bench_v1.json
```

Human gate:
- decide which variants to cache based on measured latency.

---

## Stage One — Reproducibility + manifests (versioning)
### Step 12 — Dataset fingerprinting
Create `scripts/stage1/21_fingerprint_dataset.py`:
- compute sha256 of `manifest_v1.json` as `manifest_fingerprint`
- write:
  - `data/processed/deepdetect_2025/manifest_fingerprint_v1.txt`
  - `data/processed/deepdetect_2025/run_provenance_v1.json` containing:
    - git commit hash
    - python version
    - torch/torchvision versions
    - seed
    - kaggle handle
    - timestamp

Run:
```bash
python scripts/stage1/21_fingerprint_dataset.py \
  --manifest data/processed/deepdetect_2025/manifest_v1.json \
  --out data/processed/deepdetect_2025/run_provenance_v1.json
```

### Minimal metadata schema (must exist for every sample)
Required fields in manifest:
- sample_id (string)
- relpath (string)
- abspath (string)
- sha256 (string)
- bytes (int)
- label (real/fake)
- source_hint (string or null)

Optional fields (later scripts may add):
- width, height, mode, ext
- jpeg_qsig
- phash
- duplicate_group_id
- clip_embed_idx

---

## Stage One — Failure modes + assertions checklist (automate)
### Automated checks to run after each major step
1) After download:
- raw image count > 0
- directory not empty

2) After manifest:
- all sample_ids unique
- sha256 present for all
- label present for >99.5% (or block)

3) After audit:
- decode_failed <= 0.5%
- warn on label-format leakage

4) After duplicates:
- no duplicate group spans labels (warn and inspect)
- group sizes reasonable (no accidental mega-group)

5) After bias clustering:
- if cluster purity extreme, flag bias risk

6) After splits:
- every sample assigned exactly once
- no group overlap across splits
- class ratios stable

### Enforced human verification gates (STOP points)
- Gate A: folder mapping correctness
- Gate B: audit indicates no trivial leakage
- Gate C: duplicate thresholds are sane
- Gate D: bias risk decision recorded
- Gate E: canonical split chosen and justified

---

## Done criteria (Stage One complete)
Stage One is complete when these exist and pass checks:
- `manifest_v1.json` + `manifest_fingerprint_v1.txt`
- `audit_v1.parquet` + `audit_summary_v1.json` + assertions pass
- `dupes_v1/` outputs + thresholds verified
- `bias_v1/` outputs + gate D recorded
- `splits/canonical.json` + split assertions pass
- `src/dataset/` modules implemented (dataset, preprocess, features, repro)
- provenance file exists

Only then proceed to Stage Two (training baselines).

```

## Why this AGENTS.md is structured this way

The design choices in `AGENTS.md` aren’t “nice-to-haves”; they are protective scaffolding against known high-accuracy traps in fake-image detection:

- **Audit + leakage checks are mandatory** because detectors can learn dataset-specific artifacts (format, resizing pipelines, compression choices) rather than general fake cues; modern work highlights sensitivity to dataset and augmentation choices and notes that some earlier detectors do not generalize to newer models. citeturn2search6turn2search0  
- **Robustness to laundering** is a central issue; Cozzolino et al. explicitly study robustness under impaired/laundered conditions and show meaningful AUC swings—your Stage One audits and splits are what make later robustness tests interpretable (rather than confounded by trivial dataset shortcuts). citeturn2search9turn2search5  
- **Source-/generator-disjoint evaluation** is the cleanest way to approximate “unseen generator” generalization. Park & Owens emphasize that the key challenge is detecting images from previously unseen generators and that training set diversity is a major obstacle. citeturn2search6turn2search10

## Integration deltas you should expect when adding this file to an existing repo

If your repo already has partial code, the most common integration edits are:

- A single canonical config point (e.g., `configs/deepdetect_2025.yaml`) that defines:
  - paths (`data/raw/...`, `data/processed/...`)
  - seed
  - accepted extensions
  - label mapping rules
- Consolidating existing scripts into `scripts/stage1/` and making them idempotent (no silent overwrites).
- Adding a `src/dataset/` package and converting ad-hoc notebook logic into importable modules:
  - `deepdetect.py`, `preprocess.py`, `features.py`, `repro.py`
- Adding a “do-not-commit-data” `.gitignore` update.
- Adding a one-liner “stage one runner” (optional but helpful) such as:
  - `make stage1` or `python scripts/stage1/run_all.py` that calls each script with the same manifest path.

If you want this to be truly production-grade, add CI checks that validate:
- `manifest_v1.json` schema
- splits are disjoint by group
- audit decode failure rate under threshold
- provenance file exists for any experiment run