# Dataset
Summary of the Time Series Datasets Used in Our Experiments

| Dataset | Domain       | Features | Time Steps | Sampling Interval | Original Missing Rate | Sequence Length |
|---------|--------------|----------|------------|-------------------|-----------------------|-----------------|
| ETTm1   | Energy       | 7        | 96         | 15 minutes        | 0%                    | 96              |
| Weather | Meteorology  | 21       | 144        | 10 minutes        | 0.017%                 | 144             |
| METR-LA | Traffic      | 207      | 288        | 5 minutes         | 8.6%                   | 24              |
| Yeast   | Biology      | 7        | 185        | 5 minutes         | 0%                    | 186             |

The METR-LA data files are:

- `metr_la/metr-la.h5`: Original source data.
- `metr_la.pkl`: Preprocessed data used by the experiments.
