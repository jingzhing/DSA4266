# DeepDetect Swin + EfficientNet Ensemble Pipeline

This is a rebuilt training pipeline for DeepDetect with a cleaner split design and a one-command runner.

## Dataset layout

Expected dataset root:

- `data/deepdetect-2025/ddata/train`
- `data/deepdetect-2025/ddata/test`

Each split must contain:

- `real/`
- `fake/`

You can also set `DEEPDETECT_DATA_DIR` to point directly at the `ddata` folder.

## Split design

The training folder is split once and saved to JSON:

- `train`
- `val_select` for checkpoint selection and early stopping
- `val_tune` for threshold tuning and ensemble search
- official `test` is only used after the models are frozen

This avoids reusing the same validation subset for everything.

## Models

- Swin Tiny at 224 px
- EfficientNet-B3 at 300 px

## Run the full pipeline

```bash
python run_all.py --workdir .
```

Or with a specific Python executable:

```bash
python run_all.py --workdir . --python "C:\path\to\venv\Scripts\python.exe"
```

## Solo runs

```bash
python train_single.py --model swin
python train_single.py --model efficientnet
python test_single.py --model swin
python test_single.py --model efficientnet
python ensemble_posthoc.py
```

## Notes

- `num_workers=2` by default
- test is not evaluated during training
- thresholds are tuned only once after the best checkpoint is chosen
- ensemble weights are searched only on `val_tune`
