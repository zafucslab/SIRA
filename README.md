# SIRA

*Multi-step soil moisture forecasting via sparse interaction and residual adaptation* (SIRA). This repository contains the original station workbooks and the code for preprocessing, training and evaluating SIRA.

The four components use the names in the manuscript.

| Component | Manuscript name | Source |
| --- | --- | --- |
| LORM | Lag-based Observation-preserving Reconstruction with an Observation Mask | `data_provider/lorm.py` |
| PCFB | Progressive Cascaded Forecasting Backbone | `models/pcfb.py` |
| DARA | Dynamic Adaptive Residual Adapter | `models/dara.py` |
| SGR | Sparse Granger Residual | `models/sira.py`, which assembles the full model and adds SGR |

## Quick start

From the repository root, use Bash and a Python 3.10 environment.

```bash
python -m pip install -r requirements.txt
bash scripts/prepare.sh
bash scripts/tonglu.sh 2021
```

Each study site has a script under `scripts/`. Pass the seed as its first argument.

## Repository layout

- `exp/SIRA.py` is the training and evaluation entry point. The experiment scripts invoke it with `python -m exp.SIRA`.
- `data_provider/` standardizes the workbooks, reconstructs missing values with LORM, and loads masked forecast windows.
- `exp/` implements training, validation, checkpoint selection and test evaluation.
- `models/` contains the SIRA model, its backbone, relationship module and residual adapters.
- `layers/` contains reversible instance normalization and the prediction backbone blocks.
- `utils/` provides time features, metrics and training utilities.
- `scripts/` contains data preparation, two-stage training and seven station entry points.
- `dataset/` contains the original Excel workbooks with lowercase English station filenames (for example, `tonglu.xlsx`).
- `graphs/` contains the shared prior (`agriculture_prior.csv`) and one relationship matrix per study site (for example, `tonglu.csv`).

## Data

The original workbooks are in `dataset/`. The seven sites used in the paper are Dinghai (定海), Kaihua (开化), Tonglu (桐庐), Chun’an (淳安), Pan’an (磐安), Xihu (西湖) and Longyou (龙游). The original archive also contains Liandu (莲都), whose 42 daily records are shorter than the 192-day input window.

Each site is processed separately. The inputs are daily air temperature, rainfall, wind speed, soil temperature and moisture at both 5 and 20 cm, solar radiation, relative humidity and sunshine duration. The forecast target is 5 cm soil moisture. The preprocessing scripts select the study period, retain observation masks, split chronologically (70%/10%/20%) and fit the reconstruction parameters on the training period. The supplied agricultural prior is used by LORM. The supplied site-specific relationship matrices are those used in the forecasting experiments.

## Experiment protocol

Preparation reads the Excel workbooks, standardizes dates and units, and writes the reconstructed values and observation masks for all seven sites to `dataset/processed/`. The training command uses 192 days of history and forecasts the next 24 days. Stage one trains PCFB and DARA jointly. Stage two initializes from the selected stage-one checkpoint and trains SGR. Each stage uses validation for checkpoint selection. The final command in the script evaluates the selected stage-two model on the held-out test period.

The shared script `scripts/train.sh` takes the site and seed as its two arguments. The paper uses seeds 2021, 2022 and 2023. To run all settings after preparing the data:

```bash
for station in chunan dinghai kaihua longyou panan tonglu xihu; do
  for seed in 2021 2022 2023; do
    bash scripts/train.sh "$station" "$seed"
  done
done
```

Training uses the first visible GPU when CUDA is available. For a CPU run use `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 bash scripts/train.sh tonglu 2021`. Each stage allows up to 10 epochs. For a short installation check use `EPOCHS=1` in a separate working copy so that it has its own output directory.

Checkpoints are saved under `checkpoints/`. Test predictions, targets and observation masks are saved under `results/` by site and seed. Each tensor is indexed by forecast origin, lead day and target channel. Test MSE and MAE are printed on the training-standardized scale over observed targets. RMSE is the square root of MSE.

## Manuscript

Guoying Wang, Peiyu Tian, Haoyu Wang, Lufeng Mo, Xiaomei Yi and Peng Wu. *Multi-step soil moisture forecasting via sparse interaction and residual adaptation*.

The forecasting backbone builds on [ReNF](https://github.com/Luoauoa/ReNF). See `LICENSE-ReNF` for its upstream license.
