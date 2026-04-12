# Stage-One Scripts (AGENTS.md alignment)

This folder is reserved for Stage-One pipeline scripts required by `AGENTS.md`:

- manifest/audit
- duplicate detection
- embedding + bias checks
- leakage-safe split building
- reproducibility/provenance fingerprinting

Current execution state:

1. Step 0/0b structure is created.
2. Step 1 ingestion is blocked because `kaggle` CLI is not installed in the current shell.

When `kaggle` CLI is available, continue with:

```bash
kaggle datasets download -d ayushmandatta1/deepdetect-2025 -p data/raw/deepdetect_2025 --unzip -o
```
