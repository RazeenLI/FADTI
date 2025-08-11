# Fourier Bias Projection for Time Series Imputation

This repository contains the implementation of our time series imputation framework based on conditional diffusion models with frequency-domain inductive biases.

## Overview
We propose a modular Fourier Bias Projection (FBP) module that supports multiple Fourier-based transforms, such as DFT, and our tailored implementations of STFT and FrSST, to enhance the modeling of stationary and non-stationary components in multivariate time series.

## Installation
```bash
pip install -r requirements.txt
```

## Dataset
All required datasets are already included in the `data/` directory for reproducibility.

Summary of the Time Series Datasets Used in Our Experiments

| Dataset | Domain       | Features | Time Steps | Sampling Interval | Original Missing Rate | Sequence Length |
|---------|--------------|----------|------------|-------------------|-----------------------|-----------------|
| ETTm1   | Energy       | 7        | 96         | 15 minutes        | 0%                    | 96              |
| Weather | Meteorology  | 21       | 144        | 10 minutes        | 0.017%                 | 144             |
| METR-LA | Traffic      | 207(9)   | 288        | 5 minutes         | 8.6%                   | 24              |
| Yeast   | Biology      | 7        | 185        | 5 minutes         | 0%                    | 186             |


## Usage

### 1. Test all methods (background)
```bash
nohup ./run.sh > run2.log 2>&1 &
```
This will run all supported models with default settings defined in run.sh.

### 2. Run FADTI directly (background)
```bash
nohup ./run_fadti.sh > run2.log 2>&1 &
```
This will execute the FADTI model with the predefined configuration in run_fadti.sh,
redirecting both stdout and stderr to run2.log while running in the background.

### 3. Run a standard method
```bash
python run_all.py --model "csdi" --data "ett" --nsample 100 --device "cuda:0" --nfold 5 --misspattern "point" --missrate 0.1 
```
Replace `csdi` with the desired model name (e.g., `timesnet`, `timemixer`, `saits`).

### 4. Run a specific FADTI manually
```bash
python run.py --model "fadti" --ffttype "dft" --timetype "point" --data "ett" --nsample 100 --device "cuda:0" --nfold 0 --misspattern "point" --missrate 0.1
```
Arguments

- `--model`: Model name (Only Fadti)
- `--ffttype`: Fourier transform type (dft, stft, frsst)
- `--timetype`: Time masking type (point, time)
- `--data`: Dataset name (ett, weather, metrla, yeast)
- `--nsample`: Number of samples for imputation
- `--device`: Device to run on (cpu, cuda:0, etc.)
- `--nfold`: Number of folds for cross-validation
- `--misspattern`: Missing pattern (point, time)
- `--missrate`: Missing rate (0.1, 0.5, etc.)


<!-- ## Results
- 可以放一张表格（比如你之前的 dataset table 或性能对比）
- 如果匿名投稿，结果只需要展示实验配置和输出格式，不要放真实论文图 -->

## License
For review purposes only.