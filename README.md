# FADTI: Fourier and Attention Driven Diffusion for Multivariate Time Series Imputation

Accepted at the 2026 IEEE International Conference on Data Mining (ICDM 2026).

[Paper (arXiv)](https://arxiv.org/abs/2512.15116)

This repository contains the implementation of our time series imputation
framework based on conditional diffusion models with frequency-domain inductive
biases.

## Overview

We propose a modular Fourier Bias Projection (FBP) module that supports multiple
Fourier-based transforms, including DFT and our tailored implementations of STFT
and FSST. The module is designed to improve the modeling of stationary and
non-stationary components in multivariate time series.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Datasets

The datasets required by the provided experiments are included in the `data/`
directory.

| Dataset | CLI identifier | Domain | Features | Original time steps | Sampling interval | Original missing rate | Sequence length |
|---------|----------------|--------|----------|---------------------|-------------------|-----------------------|-----------------|
| ETTm1 | `ett` | Energy | 7 | 96 | 15 minutes | 0% | 96 |
| Weather | `weather` | Meteorology | 21 | 144 | 10 minutes | 0.017% | 144 |
| METR-LA | `metr_la` | Traffic | 207 | 288 | 5 minutes | 8.6% | 24 |
| E. coli | `ecoli` | Biology | 7 | 185 | 5 minutes | 0% | 185 |

Use `metr_la` as the command-line identifier for the METR-LA dataset. See
[`data/README.md`](data/README.md) for the files included with the repository.

## Quick Start

Run a single FADTI experiment:

```bash
python3 run.py \
  --model fadti \
  --ffttype dft \
  --timetype attn \
  --data ett \
  --device cuda:0 \
  --nfold 0 \
  --misspattern point \
  --missrate 0.1 \
  --nsample 100
```

Run a single baseline experiment:

```bash
python3 run_all.py \
  --model csdi \
  --data ett \
  --device cuda:0 \
  --nfold 0 \
  --misspattern point \
  --missrate 0.1 \
  --nsample 100
```

Set `--device cpu` to run without a GPU. Diffusion-model training and evaluation
may be substantially slower on CPU.

## Entry Points

- `run.py` runs FADTI and exposes the Fourier transform and time-processing
  options.
- `run_all.py` runs FADTI or a baseline through the shared model interface. When
  `fadti` is selected here, it uses the default DFT and attention configuration
  defined in the model loader.
- `run_fadti.sh` runs the provided FADTI ablation grid.
- `run.sh` runs the provided baseline experiment grid.

## Supported Models

The following identifiers are accepted by the model loader used by `run_all.py`:

| Identifier | Method |
|------------|--------|
| `mean` | Mean imputation |
| `median` | Median imputation |
| `knn` | K-nearest-neighbor imputation |
| `csdi` | CSDI |
| `csdi_ori` | Original CSDI implementation |
| `fadti` | FADTI with the default loader settings |
| `saits` | SAITS |
| `brits` | BRITS |
| `timesnet` | TimesNet |
| `mtsci` | MTSCI |
| `timemixer` | TimeMixer |
| `timemixerpp` | TimeMixer++ |
| `ssdts` | SSD-TS |

## Arguments

### Common arguments

- `--data`: Dataset identifier: `ett`, `weather`, `metr_la`, or `ecoli`.
- `--device`: Computation device, such as `cpu`, `cuda:0`, or `cuda:1`.
- `--nsample`: Number of imputation samples generated during evaluation.
- `--nfold`: Cross-validation fold index from 0 to 4.
- `--missrate`: Artificial missing rate, such as `0.1` or `0.5`.
- `--misspattern`: Artificial missingness pattern. The implementation accepts
  `point`, `time`, and `block`; the provided batch scripts evaluate `point` and
  `time`.
- `--modelfolder`: Existing directory name under `save/` from which to load
  `model.pth`. If omitted, the selected model is trained before evaluation.

### FADTI-specific arguments

These arguments are exposed by `run.py`:

- `--model`: Must be `fadti`.
- `--ffttype`: Fourier transform type: `none`, `dft`, `stft`, or `frsst`.
- `--timetype`: Time-processing layer: `attn` or `conv`.

### Baseline selection

Use `--model` with `run_all.py` to select one of the identifiers in the
[Supported Models](#supported-models) table.

### SSD-TS

SSD-TS can be run through the shared baseline entry point:

```bash
python3 run_all.py \
  --model ssdts \
  --data ett \
  --device cuda:0 \
  --nfold 0 \
  --misspattern point \
  --missrate 0.1 \
  --nsample 100
```

In addition to the core project dependencies, SSD-TS requires `einops`,
`mamba_ssm`, and `causal_conv1d` in the execution environment. These packages
are only imported when SSD-TS is selected. SSD-TS is not included in the default
`run.sh` experiment grid.

Install these optional dependencies with:

```bash
python3 -m pip install -r requirements-ssdts.txt
```

## Batch Experiments

### FADTI ablation grid

```bash
nohup ./run_fadti.sh > run_fadti.log 2>&1 &
```

With its current settings, this script runs 256 sequential experiments across
four Fourier options, two time-processing layers, four datasets, two folds, two
missing rates, and two missing patterns.

### Baseline experiment grid

```bash
nohup ./run.sh > run_all.log 2>&1 &
```

With its current settings, this script runs 880 sequential experiments across 11
models, four datasets, five folds, two missing rates, and two missing patterns.
Both batch scripts may therefore take a substantial amount of time.

The experiment grids can be changed by editing the arrays and `nsample` value at
the top of each script. The scripts use `cuda:0` and `python3` by default. These
can be overridden without editing the scripts:

```bash
DEVICE=cuda:1 ./run.sh
DEVICE=cpu PYTHON_BIN=/path/to/python3 ./run_fadti.sh
```

## Outputs

Every invocation creates a timestamped directory under `save/`. FADTI runs use:

```text
save/fadti_<ffttype>_<timetype>_<dataset>_<pattern>_<rate>_<nsample>_<timestamp>/
```

Runs through `run_all.py` use:

```text
save/<model>_<dataset>_<pattern>_<rate>_<timestamp>/
```

Depending on the model and save strategy, the directory may contain:

- `config.json`: Merged model and dataset configuration.
- `model.pth`: Saved model parameters.
- `result_train_valid.pk`: Training and validation history.
- `generated_outputs_nsample<N>.pk`: Generated imputation samples.
- `result_nsample<N>.pk`: Evaluation metrics and related results.

When `--modelfolder <folder-name>` is provided, the program loads
`save/<folder-name>/model.pth`. A new timestamped directory is still created for
the current configuration and evaluation outputs.

## Repository Structure

```text
FADTI/
├── config/       Model and dataset configurations
├── data/         Original and preprocessed datasets
├── dataset/      Dataset loading, splitting, and masking
├── model/        FADTI and baseline implementations
├── nn/           Shared neural-network components and metrics
├── utils/        Data-loader and model factories
├── run.py        Configurable FADTI entry point
├── run_all.py    Shared baseline entry point
├── run_fadti.sh  FADTI ablation grid
└── run.sh        Baseline experiment grid
```

## Reproducibility Notes

- Dataset configurations define the seed used for data splitting and artificial
  masking.
- Cross-validation folds are indexed from 0 to 4.
- Each run saves its merged model and dataset configuration as `config.json`.
- Reproducing the same results requires the same data, configuration, dependency
  environment, and random settings.
- Small numerical differences may occur across GPU hardware and software stacks.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{li2025fadti,
  author  = {Runze Li and Hanchen Wang and Wenjie Zhang and Binghao Li and
             Yu Zhang and Xuemin Lin and Ying Zhang},
  title   = {FADTI: Fourier and Attention Driven Diffusion for Multivariate
             Time Series Imputation},
  journal = {CoRR},
  volume  = {abs/2512.15116},
  year    = {2025},
  doi     = {10.48550/arXiv.2512.15116}
}
```

## License

The original software in this repository is released under the MIT License.
See [LICENSE](LICENSE) for details. Dataset files remain subject to their
respective upstream terms.
