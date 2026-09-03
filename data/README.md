# Datasets

This directory contains the time-series datasets used in our experiments. The
table below summarizes the data after preprocessing.

| Dataset | Domain       | Features | Time Steps | Sampling Interval | Original Missing Rate | Sequence Length |
|---------|--------------|----------|------------|-------------------|-----------------------|-----------------|
| ETTm1   | Energy       | 7        | 96         | 15 minutes        | 0%                    | 96              |
| Weather | Meteorology  | 21       | 144        | 10 minutes        | 0.017%                 | 144             |
| METR-LA | Traffic      | 207      | 288        | 5 minutes         | 8.6%                   | 24              |
| E. coli | Biology      | 7        | 185        | 5 minutes         | 0%                    | 185             |

The METR-LA data files are:

- `metr_la/metr-la.h5`: Original source data.
- `metr_la.pkl`: Preprocessed data used by the experiments.

## Sources and Download Instructions

### ETTm1

- **Source:** The Electricity Transformer Temperature (ETT) dataset released
  with the Informer project.
- **Official repository:** https://github.com/zhouhaoyi/ETDataset
- **Download:** Download `ETTm1.csv` from the `ETT-small` directory in the
  official repository and place it at `data/ETTm1.csv`.
- **Files in this repository:** `ETTm1.csv` is the source CSV file and
  `ettm1.pk` is the preprocessed file used by the experiments.
- **Terms:** The official ETT repository distributes the dataset under the
  Creative Commons Attribution-NoDerivatives 4.0 International license
  (CC BY-ND 4.0). Users should review the upstream license before using or
  redistributing either the source data or derived files:
  https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE

### Weather

- **Source:** Weather observations from the Weather Station of the Max Planck
  Institute for Biogeochemistry in Jena, Germany.
- **Official data page:** https://www.bgc-jena.mpg.de/wetter/weather_data.html
- **Download:** Obtain the 2020 observations from the official data page. The
  benchmark CSV used by this project contains 21 meteorological variables at
  10-minute intervals and should be placed at `data/weather.csv`.
- **Files in this repository:** `weather.csv` is the source CSV file and
  `weather.pk` is the preprocessed file used by the experiments.
- **Terms:** No explicit redistribution license for this packaged benchmark
  file has been identified on the official data page. Users should review the
  provider's current terms before redistributing the source or processed data.

### METR-LA

- **Source:** Traffic-speed measurements from 207 loop detectors in the Los
  Angeles County highway network.
- **Download and preparation instructions:**
  https://github.com/liyaguang/DCRNN#data-preparation
- **Download:** Follow the METR-LA instructions in the DCRNN repository and
  place the downloaded HDF5 file at `data/metr_la/metr-la.h5`.
- **Files in this repository:** `metr_la/metr-la.h5` is the downloaded source
  file and `metr_la.pkl` is the preprocessed file used by the experiments.
- **Terms:** The DCRNN software repository uses the MIT License, but that
  software license does not by itself establish redistribution rights for the
  underlying traffic data. Users should consult the original data provider's
  terms before redistributing the dataset.

### E. coli

- **Source:** Single-cell *E. coli* microscopy time series from the study
  "Deep model predictive control of gene expression in thousands of single
  cells."
- **Official record:** https://doi.org/10.5281/zenodo.8114649
- **Download:** Download `datasets.zip` from the Zenodo record. The Nature
  Communications article confirms that the study datasets and processed data
  are deposited under accession 8114649.
- **Files in this repository:** `ecoli_set2.pkl` is the source dataset and
  `ecoli.pk` is the preprocessed file used by the experiments.
- **Reference:** https://doi.org/10.1038/s41467-024-46361-1
- **Terms:** The data are publicly available through Zenodo. The Zenodo record
  should be consulted for its current terms of use.

## Reproducibility Notes

- Keep the filenames and paths above unchanged because the data loaders expect
  these locations.
- The `.pk` and `.pkl` files are serialized Python objects. Load them only when
  they come from a trusted source.
- Dataset terms are separate from this repository's software license. A code
  license does not automatically grant permission to redistribute datasets.
