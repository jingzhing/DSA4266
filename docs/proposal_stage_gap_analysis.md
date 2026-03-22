# Proposal Stage Gap Analysis (DSA4266 Group 1)

This document maps the current repository to the proposal stages and identifies what is still missing.

## Stage Coverage Matrix

| Proposal Stage | Current Status | Evidence in Repo | Gaps / Missing Work |
|---|---|---|---|
| Data Collection (DeepDetect 2025) | Implemented | `pipeline/stages.py` (`run_setup`), `configs/pipeline.yaml` (`data.dataset_id`) | Alternative datasets listed in proposal are not integrated into pipeline ingestion. |
| Data Preprocessing (resize/normalize/split/pipeline) | Implemented | `pipeline/stages.py` (`run_prepare`), model-specific loaders in `pipeline/models.py` | Current pipeline does not include explicit cross-domain balancing controls beyond basic class structure. |
| Optional Data Enrichment | Implemented | `pipeline/video.py`, `pipeline/stages.py` (`with_video` path) | No curated policy for trusted source lists, filtering quality metrics beyond Laplacian variance. |
| Transfer Learning (baseline finetuning) | Implemented | Swin + EfficientNet train flows in `pipeline/models.py` | Automated experiment tracking for linear probing vs progressive unfreezing is limited. |
| Linear Probing | Partial | EfficientNet supports freeze flag; Swin can be extended | No first-class CLI/config strategy matrix runner for linear probing across both model families. |
| Progressive Unfreezing | Partial | EfficientNet has freeze behavior scaffold | No staged unfreeze scheduler implemented for Swin/EfficientNet as proposal method. |
| PEFT (Transformer) | Missing | N/A | No LoRA/adapters or other PEFT implementation in Swin path. |
| Architecture Modification | Partial | EfficientNet head adapts classifier layer | No explicit architecture-search or modular plugin path for systematic head/backbone variants. |
| Frequency-Domain CNN (FFT/DCT + CNN) | Missing (as canonical stage) | Legacy exploratory notebook moved to `old/` | No active pipeline stage/model integrating FFT/DCT features with shared CLI and artifacts. |
| Swin Transformer Path | Implemented | `pipeline/models.py` (`swin` handlers) | Needs planned fine-tuning strategy matrix and robustness benchmarking. |
| Evaluation (Accuracy/Precision/Recall/F1/CM) | Implemented | `pipeline/metrics.py`, `run_eval` artifact outputs | Proposal-specific reporting templates (tables by risk profile, error-cost emphasis) not yet automated. |
| Robustness vs Compression/Post-processing/Adversarial | Missing | N/A | No dedicated robustness evaluation stage or perturbation benchmark in active pipeline. |
| Generalization Across Unseen Generators | Missing/Partial | Single dataset baseline path | No held-out-generator protocol and no multi-dataset generalization harness yet. |

## What Is Already Strong

1. Canonical, runnable CLI pipeline with explicit stages.
2. Shared prepared dataset contract for both Swin and EfficientNet.
3. Preflight checks and standardized artifacts for reproducibility.
4. Optional video frame enrichment integrated without polluting immutable raw dataset.

## Highest-Priority Missing Stages (from proposal intent)

1. **Frequency-domain canonical model path**
   - Add an active model/stage that consumes FFT/DCT representations, not only old notebook code.
2. **Transfer-learning strategy matrix**
   - First-class experiment modes for linear probing and progressive unfreezing for both models.
3. **PEFT for Swin**
   - Add parameter-efficient fine-tuning methods (e.g., LoRA/adapters) with config+CLI support.
4. **Robustness evaluation stage**
   - Add perturbation benchmarks (compression, blur/noise, adversarial transforms) and artifacted reports.
5. **Generalization protocol**
   - Introduce held-out generator evaluation or alternative datasets from proposal list.

## Suggested Next Iteration Plan

1. Add `models.freq_cnn` section to YAML and implement `freq_cnn` handlers in `pipeline/models.py`.
2. Add `training.mode` options: `linear_probe`, `progressive_unfreeze`, `full_finetune`.
3. Add `evaluation.robustness` config and a new `robust-eval` CLI stage.
4. Add experiment aggregation report stage (`report`) to summarize metrics across model/mode combinations.
