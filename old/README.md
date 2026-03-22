# old/

This folder stores original/legacy/unused assets moved out of the active pipeline code path.

Contents currently include:

- `old/models/swin/v1/` - legacy Swin training/testing scripts (pre-pipeline architecture).
- `old/models/efficientnet/EfficientNet_Testing.ipynb` - original EfficientNet notebook exploration.
- `old/scripts/data_aug.ipynb` - legacy augmentation notebook.
- `old/results/` - historical result images from earlier runs.

Active development should use:

- `pipeline/` for implementation
- `configs/pipeline.yaml` for configuration
- `tests/` for validation

Do not add new production logic under `old/`.
