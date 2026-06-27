# DTAN

DTAN is a bi-encoder Transformer framework for accurate antigenic prediction of influenza A viruses.

The overall framework of DTAN is shown below.



<img width="1525" height="1168" alt="dtan_framework" src="https://github.com/user-attachments/assets/cb9c1079-c3bf-4d3c-9273-e10735798a87" />



The current repository supports four major tasks:
- reproduction of DTAN and ablation experiments for three subtypes
- runtime comparison between DTAN and NoDAE under different data fractions
- site-importance analysis
- inference and export of results for downstream antigenic mapping

## Project Structure

Commonly used files and directories:

- `main.py`: main entry point for training, validation, and testing
- `main_time.py`: entry point for runtime comparison
- `src/`: core code for data handling, feature extraction, training, evaluation, and importance analysis
- `result/joblib/`: cross-validation outputs used for plotting
- `result/importance/`: exported attention or importance scores
- `saved_models/`: model weights and encoders for inference
- `download_ret/`: inference input and output files

## Environment

Recommended environment:

- Python 3.10.13
- biopython 1.84
- joblib 1.4.2
- pandas 2.2.2
- numpy 1.26.1
- scikit-learn 1.7.0
- scipy 1.15.3
- torch 1.11.0+cu113
- tqdm 4.66.1
- xgboost 2.0.3

Install example:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- `requirements.txt` pins the main package versions.
- The project also depends on `loguru` and `aaindex`.

## Main Arguments

Common arguments in `main.py`:

- `--seq_path`: sequence file path
- `--anti_path`: antigenic data file path
- `--subtype`: `H1`, `H3`, or `H5`
- `--label`: `NHT_distance` is the recommended setting
- `--valid_type`: `titer`, `train`, or `test`
- `--model`: `full`, `cnn`, `no_encoder`, or `no_meta`
- `--low`: `ae`, `none`, `onehot`, or `pca`
- `--ae_dim`: latent feature dimension
- `--epochs`: number of training epochs
- `--im_lambda`: weight for the importance loss term
- `--im_epoch`: number of pretraining epochs for importance learning
- `--device`: device identifier such as `cuda:0`

Use `--epochs`, not `--epoch`.

## Quick Start

Example command for the H1 DTAN setting (`full + ae`):

```bash
python main.py \
  --seq_path data/dataset0722/H1N1_N2_HA_syed.csv \
  --anti_path data/dataset0722/H1N1_N2_HI_syed.csv \
  --subtype H1 \
  --label NHT_distance \
  --valid_type titer \
  --n_jobs 30 \
  --low ae \
  --ae_dim 64 \
  --epochs 20 \
  --model full \
  --im_lambda 3e-5 \
  --im_epoch 10 \
  --device cuda:0
```

After the run:
- logs are written under `log/<date>/<model>_<low>/<subtype>/`
- joblib outputs are written under `result/joblib/<model>_<low>/`

## Reproducing DTAN and Ablations

Run `main.py` to train DTAN or one of its ablation variants:

```bash
python main.py \
  --seq_path <HA_csv> \
  --anti_path <HI_csv> \
  --subtype <H1|H3|H5> \
  --label NHT_distance \
  --valid_type titer \
  --n_jobs 30 \
  --low <ae|none|onehot> \
  --ae_dim <64|20> \
  --epochs <20|300> \
  --model <full|cnn|no_encoder|no_meta> \
  --im_lambda <3e-5|5e-5> \
  --im_epoch <10|25> \
  --device <cuda:x>
```

Typical settings:
- DTAN: `model=full`, `low=ae`
- DTAN-CNNEnc: `model=cnn`, `low=ae`
- DTAN-NoEnc: `model=no_encoder`, `low=ae`
- DTAN-NoMeta: `model=no_meta`, `low=ae`
- DTAN-NoDAE: `model=full`, `low=none`
- DTAN-OneHot: `model=full`, `low=onehot`

## Runtime Comparison

Run `main_time.py` to compare the runtime of DTAN and NoDAE under different training-data fractions:

```bash
python main.py \
  --seq_path <HA_csv> \
  --anti_path <HI_csv> \
  --subtype <H1|H3|H5> \
  --label NHT_distance \
  --valid_type titer \
  --n_jobs 30 \
  --low <ae|none> \
  --ae_dim 64 \
  --epoch <20|300> \
  --model full \
  --percentage <0.33333333|0.66666666|1.0> \
  --im_lambda <3e-5|5e-5> \
  --im_epoch <10|25> \
  --device <cuda:x>
```

Notes:
- the key parameter is `--percentage`, for example `0.33333333`, `0.66666666`, or `1.0`

## Site-Importance Analysis

Suggested workflow:

1. In `src/utils.py`, uncomment the line inside `only_train` that exports attention scores.
2. Train with `main.py` using `--valid_type train` for each subtype.
3. Exported scores are saved under `result/importance/<date>/<subtype>_<time>/`.
4. Update subtype and input paths in `src/importance.py`.
5. Run:

```bash
python src/importance.py
```

The resulting files can then be used in the existing PyMOL workflow.

## Inference and Export

### Required model files

For H1, make sure the following files are available:

- `saved_models/H1/H1.pt`
- `saved_models/H1/H1_at_ohe.joblib`
- `saved_models/H1/H1_sr_ohe.joblib`

### Inference input

In test mode, `main.py` reads:

- `download_ret/H1_with_seq.csv`
- `download_ret/H3_with_seq.csv`
- `download_ret/H5_with_seq.csv`

Each input file should contain:
- `virus1`
- `virus2`
- `seq1`
- `seq2`

### Run inference

```bash
python main.py
--seq_path <HA_csv>
--anti_path <HI_csv>
--subtype <H1|H3>
--label NHT_distance
--valid_type test
--n_jobs 30
--low ae
--ae_dim 64
--epoch 20
--model full
--im_lambda 3e-5
--im_epoch 10
--device cuda:x
--test_file <H1_results.csv|H3_results.csv>
```

Output results can be merged with HI distances for downstream antigenic mapping.

## Output Locations

- training logs: `log/<date>/...`
- cross-validation outputs and metrics: `result/joblib/...`
- importance scores: `result/importance/...`
- inference results: `download_ret/*_results.csv`
- figure outputs: `result/figure/...`

## Troubleshooting

### `unrecognized arguments: --epoch`

Use `--epochs` instead.

### `--test_file` is ignored in test mode

The current test branch of `main.py` reads fixed files under `download_ret/<subtype>_with_seq.csv`. The `--test_file` argument is not actively used.

### Logs do not appear in the terminal

Logs are written to files under `log/` by design, and console output is disabled in the current implementation.
