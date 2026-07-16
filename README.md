# LSTM Activations Analysis

Analysis of fMRI signals processed through recurrent (LSTM) models. The pipeline parcellates
Human Connectome Project (HCP) 7T movie-watching and resting-state scans into ROI/network time
series, trains an LSTM to classify movie clips from those time series, and then studies the
model's internal **activations** — how the network represents clips versus nearby rest, via
relational coding, connectivity, and correlation statistics.

> This repository accompanies ongoing, unpublished research. It describes the algorithms and
> how to run them; it intentionally contains no findings or citations.

## What it does

1. **Parcellation** — raw CIFTI movie/rest data → Schaefer-2018 ROI (300) / 7-network time series.
2. **Model training** — an LSTM clip-classifier over the ROI/network time series.
3. **Activation analysis** — extract the model's activations and compare clip vs. rest structure:
   - **relational coding** — task-vs-rest relational distance per clip;
   - **connectivity** — per-network connectivity matrices;
   - **correlation pipelines** — whole-brain / per-network activation correlations.

## Layout

```
settings.py               # paths + LSTM_DATA_DIR / LSTM_OUTPUT_DIR (.env) overrides
enums.py                  # Mode / Network / DataType
cli.py                    # command-line interface over the main workflows
parcellation/             # raw CIFTI -> ROI/network time series (own argparse script)
model_training/           # LSTM model, dataloader, training, inference, cc/torch utils
relational_coding/        # task-vs-rest relational-coding distances
fmri_connectivity_matrices/  # network connectivity matrices
statistical_analysis/     # correlation pipelines + math/matrix helpers
mappings/                 # clip/rest ordering and subject mappings
visualizations/           # plotting helpers
supporting_functions.py   # pickle / csv I/O helpers
tests/                    # pytest suite (algorithms, config, CLI)
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Version pins mirror the
validated environment (Python 3.10, torch 1.12); results depend on them, so avoid bumping.

```bash
uv sync                 # create the environment from pyproject.toml + uv.lock
uv run pytest           # run the test suite
```

## Configuration

Data and outputs live under the repo by default. `settings.py` reads two roots from the
environment or a local `.env`; copy the template and edit it:

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `LSTM_DATA_DIR` | repo root | inputs: `fmri_data`, `cifti` parcellation, `mappings` |
| `LSTM_OUTPUT_DIR` | repo root | outputs: `saved_models`, `activation_matrices`, correlation/connectivity results, `figures` |

A full run expects the preprocessed fMRI tables and trained models to already exist under
those paths.

## Running

```bash
uv run python -m cli train --mode rest_between --epochs 50
uv run python -m cli infer --model-mode combined --inference rest_between
uv run python -m cli correlation --scope net
uv run python -m cli relational-coding --shuffle
uv run python -m cli connectivity
```

`python -m cli --help` (or `<command> --help`) lists all options. The underlying functions
remain importable from their modules for use in notebooks/scripts.

Raw-data parcellation keeps its own script:

```bash
uv run python parcellation/parcellation.py --input-data <cifti> --output-data <roi> --roi 300 --net 7
```

## Testing

```bash
uv run pytest
```

The suite covers the pure algorithm/helper code (analytic checks and characterization on
synthetic data), the settings/env overrides, and the CLI wiring. It uses only synthetic
fixtures, so it runs without the HCP data on disk.
