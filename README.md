# FE-AE: Fractal–Entropy-Driven Adaptive Ensemble for Concept Drift in Intrusion Detection

Code for the paper *Fractal–Entropy-Driven Adaptive Ensemble Learning for Concept Drift in Intrusion
Detection Systems* (A. Suresh, G. Patil, A. C. Jose).

FE-AE couples a label-free drift signal with an adaptive ensemble of Logistic Regression, Complement
Naive Bayes and Extra Trees:

* **Drift signal** (`feae/core.py`): for every window of 512 flows, a Shannon entropy on
  reference-anchored bins and a Petrosian fractal dimension on reference-mean crossings are computed on
  the first three latent coordinates. Both are compared with thresholds calibrated by bootstrap on a
  reference set. An optional error channel applies the same two statistics to the 0/1 error sequence.
* **Adaptive weighting** (`feae/core.py`, `stream`): the ensemble weights follow a sharpened exponential
  moving average of the recent accuracy of the learners and are reset when the signal fires.
* **Leakage-free pipeline** (`feae/pipeline.py`): imputer, scaler, SMOTE, PCA and the learners are fitted
  on the training partition only; the prediction for a window uses weights computed from earlier windows.

## Installation

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10–3.12.

## Data

The datasets are public and are not redistributed here. Download them and place them in `data/`
(or set `FEAE_DATA`), with the file names listed in `config.py`:

| Dataset | Files |
|---|---|
| NSL-KDD | `KDDTrain+.txt`, `KDDTest+.txt` |
| UNSW-NB15 | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| ToN-IoT (network) | `train_test_network.csv` |
| CIC-DDoS2019 | `cicddos2019/*.parquet` |
| IoT Network Intrusion Dataset | `IoT Network Intrusion Dataset.csv` |

A random sample of 60,000 flows is drawn from each dataset (`random_state=0`). The identifier, address,
timestamp and label-derived columns that are removed before training are listed in `config.py` (`EXCLUDE`).

## Reproducing the experiments

All scripts are run from the repository root. Results are written to `results/` (or `FEAE_OUT`).

| Step | Command | Paper |
|---|---|---|
| Synthetic drift streams, six River detectors, signal components | `python experiments/synthetic_benchmark.py` | Tables 5–7 |
| KS baseline, type of change, detectability bound | `python experiments/drift_typing.py` | Table 8, Eq. (3) |
| Fit the pipeline and cache the test streams | `python experiments/prepare.py` | – |
| Stationary streams, t-tests, sensitivity, label delay | `python experiments/stationary.py` | Tables 11, 12, 14 |
| Emerging attacks | `python experiments/emerging.py` | Tables 9, 10 |
| Input noise and label poisoning | `python experiments/robustness.py` | Table 15 |
| Model size and inference time | `python experiments/runtime.py` | Table 16 |
| Leakage checks (NSL-KDD) | `python experiments/leak_checks.py` | Section 5.3 |
| Figures | `python experiments/figures.py` | Figs. 2, 3 |

`prepare.py`, `stationary.py`, `emerging.py`, `robustness.py` and `runtime.py` accept dataset names
(`ToN NSL UNSW DDoS IoT`) as arguments to run a subset, e.g. `python experiments/prepare.py NSL`.
The two synthetic scripts need no dataset.

## Settings

| Symbol | Value | Meaning |
|---|---|---|
| W | 512 | window length (flows) |
| m | 3 | monitored latent coordinates |
| B | 20 | histogram bins per coordinate |
| R | 4 | windows forming the reference set |
| N_b | 200 | bootstrap windows for calibration |
| K | 4 | alarm threshold (bootstrap standard deviations) |
| δ | 0.01 | minimum rise of the error rate (error channel) |
| α | 0.7 | EWMA smoothing of the weights |
| β | 30 | sharpening exponent of the target weights |

Seeds: 11, 21, 42, 52, 84 (stationary and synthetic streams); 11, 42, 84 (emerging-attack and robustness
scenarios). These values are the defaults in `feae/core.py` and `config.py`.

## Using the drift signal on your own data

```python
import numpy as np
from feae import detect

X = np.random.default_rng(0).normal(size=(10240, 3))   # three monitored coordinates
X[5120:] *= 2.0                                        # a change of scale
print(detect(X, use_err=False))                        # one character per window: '.', 'H' (entropy) or 'F' (PFD)
```

## Notes

* The streaming evaluation is an offline sequential simulation on benchmark data; no live traffic is processed.
* Deep Feature Synthesis is applied to the PCA output with the default Featuretools configuration, which
  returns the latent coordinates unchanged for a single table; if Featuretools cannot be imported the step is skipped.
* Timings depend on the hardware.

## License

MIT, see `LICENSE`.
