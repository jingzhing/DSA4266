# DSA4266 - Unified AI-Generated Image Detection Pipeline

This repository is now organized around one canonical pipeline:

`setup -> prepare -> train -> eval -> infer`

Both `swin` (PyTorch) and `efficientnet` (TensorFlow) run through the same data and artifact contracts.

## 1. Repository Structure

- `pipeline/` - source of truth for CLI, stages, preflight checks, model runners, metrics.
- `configs/pipeline.yaml` - single config for paths, data, video enrichment, augmentation, training/eval/infer.
- `scripts/` - compatibility wrappers/utilities that call into `pipeline`.
- `requirements/` - split dependency files (`base`, `swin`, `efficientnet`, `dev`).
- `tests/` - unit and smoke tests for config, prepare, preflight, and model stage flow.
- `old/` - legacy/original/unused files moved out of active pipeline (old notebooks, legacy model scripts, old result images).

## 2. How To Run This Repository

### Install dependencies

Swin path:

```bash
pip install -r requirements/base.txt -r requirements/swin.txt
```

EfficientNet path:

```bash
pip install -r requirements/base.txt -r requirements/efficientnet.txt
```

Developer checks:

```bash
pip install -r requirements/dev.txt
```

Install everything:

```bash
pip install -r requirements.txt
```

### Run by stages (recommended)

```bash
python -m pipeline.cli setup --config configs/pipeline.yaml
python -m pipeline.cli prepare --config configs/pipeline.yaml
python -m pipeline.cli train --config configs/pipeline.yaml --model swin
python -m pipeline.cli eval --config configs/pipeline.yaml --model swin
python -m pipeline.cli infer --config configs/pipeline.yaml --model swin --input data/prepared/deepdetect-2025/test
```

Swap `--model efficientnet` for EfficientNet.

### Run everything in one command

```bash
python -m pipeline.cli run-all --config configs/pipeline.yaml --model swin
python -m pipeline.cli run-all --config configs/pipeline.yaml --model efficientnet
```

Optional video enrichment:

```bash
python -m pipeline.cli prepare --config configs/pipeline.yaml --with-video --video-url "<youtube_url>"
```

### Expected data layout

- Raw immutable dataset:
  - `data/raw/deepdetect-2025/ddata/{train,test}/{real,fake}`
- Prepared training dataset:
  - `data/prepared/deepdetect-2025/{train,val,test}/{real,fake}`
- Prepare manifest:
  - `data/prepared/deepdetect-2025/manifest.json`

### Expected run artifacts

Each run creates:

`outputs/runs/<timestamp>_<model>_<tag>/`

with:

- `config_resolved.yaml`
- `preflight_report.json`
- `data_manifest_snapshot.json`
- `model_checkpoint.*`
- `metrics.json`
- `metrics.csv`
- `predictions.csv`
- `confusion_matrix.png`

## 3. How Further Work Should Be Done

Use this workflow to extend the repository safely:

1. Add/modify configuration in `configs/pipeline.yaml` first.
2. Implement stage logic in `pipeline/stages.py` (not notebooks).
3. Implement or update model behavior in `pipeline/models.py`.
4. Add dependency changes under `requirements/` split files.
5. Add tests in `tests/` for new behavior.
6. Keep notebooks as optional references only; pipeline code is canonical.
7. Keep legacy assets in `old/` and do not reintroduce them into active paths.

For adding a new model family:

1. Add model section in YAML under `models.<new_model>`.
2. Extend `pipeline/models.py` with `train/evaluate/infer` handlers.
3. Extend CLI model choices in `pipeline/cli.py`.
4. Add preflight dependency checks in `pipeline/preflight.py`.
5. Add at least one smoke test.

## 4. Proposal Gap Analysis

A full stage-by-stage evaluation against your proposal is documented in:

[`docs/proposal_stage_gap_analysis.md`](c:/Users/aria/OneDrive/Desktop/DSA4266/DSA4266/docs/proposal_stage_gap_analysis.md)

High-level status:

- Implemented: unified data collection/setup, preprocessing pipeline, dual-model training/evaluation/inference, artifact standardization.
- Partially implemented: transfer-learning variants (basic freeze/full-finetune controls exist but not full experiment matrix automation).
- Missing: first-class frequency-domain CNN pipeline (FFT/DCT as canonical stage), PEFT for transformer, robustness stress testing (compression/adversarial), cross-dataset generalization experiments and reporting.
