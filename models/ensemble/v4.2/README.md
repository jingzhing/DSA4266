# Deepfake PyTorch Pipeline

This pipeline replaces the old joint ensemble approach with a cleaner setup:

1. Train Swin solo in PyTorch
2. Train EfficientNet-B2 solo in PyTorch
3. Save both checkpoints on the same fixed validation split
4. Run a post-hoc ensemble search on validation
5. Freeze the chosen weight and threshold
6. Evaluate once on test

## Why this is better

The old ensemble had 3 main problems:
- it did not load your previously good solo checkpoints
- it did not train the fused ensemble output directly
- it hardcoded a 50/50 probability average in validation and test

This pipeline fixes that by making the solo models the primary models and the ensemble a separate validation-driven fusion step.

## Files

- `config.py`: shared configs for Swin, EfficientNet-B2, and the post-hoc ensemble
- `data.py`: shared train/val/test split and transforms
- `models.py`: PyTorch model builders and EfficientNet partial fine-tune logic
- `metrics.py`: threshold search, balanced accuracy, weight search
- `logging_utils.py`: simple tee logger
- `train_single.py`: train either Swin or EfficientNet
- `test_single.py`: test either Swin or EfficientNet
- `ensemble_posthoc.py`: search best ensemble weight and evaluate on test

## Commands

Train Swin:
```bash
python train_single.py --model swin
```

Test Swin:
```bash
python test_single.py --model swin
```

Train EfficientNet-B2:
```bash
python train_single.py --model efficientnet
```

Test EfficientNet-B2:
```bash
python test_single.py --model efficientnet
```

Run post-hoc ensemble:
```bash
python ensemble_posthoc.py
```

## Main design choices

- EfficientNet uses B2 at 260px to stay closer to your old stronger TensorFlow setup
- Swin stays at 224px
- Both use the same shared train/val split file
- Ensemble fusion is searched on validation, then frozen for test
- Default ensemble search is done on logits, not probabilities


## Added diagnostics

- Set `diagnostics["eval_test_each_epoch"] = True` in `config.py` to see whether validation is drifting away from official test during training.
- Run `python diagnose_single.py --model swin` or `python diagnose_single.py --model efficientnet` to compare validation vs test directly.
- Diagnostics are saved under each model output folder in `diagnostics/`, including summary JSON files and top-mistake CSVs.
