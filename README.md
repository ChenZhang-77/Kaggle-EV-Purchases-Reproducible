# Predicting Electric Vehicle Purchases

Reproduction code for two final entries in [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9). The final standing was **550 of 3,576 teams**, determined by a private leaderboard ROC AUC of **0.94541**.

| Selected entry | Public AUC | Private AUC | Output |
| --- | ---: | ---: | --- |
| Two-seed LightGBM and XGBoost-CatBoost weighted rank ensemble | 0.94641 | 0.94537 | `submission_weighted.csv` |
| Five-column nonnegative logistic stacking ensemble | 0.94639 | 0.94541 | `submission_stacking.csv` |

The code trains all base models and generates out-of-fold and test predictions from source data. Both final submissions are produced in one run. It does not upload to Kaggle.

## Data

Obtain the official `train.csv`, `test.csv`, and `sample_submission.csv` from the [competition data page](https://www.kaggle.com/competitions/playground-series-s6e9/data). Obtain `EV_Adoption_and_Range_Anxiety_Dataset.csv` from the [source dataset](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety) and name it `original.csv` locally. These files are not included in Git.

Put the files in `data/`, or pass `--data-dir` pointing to a directory with the same four filenames:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
  original.csv
```

The extra dataset is used to calculate category-level statistics. It is not appended to the training set.

## Run

Python 3.12 is recommended. On macOS, LightGBM and XGBoost may require an installed OpenMP runtime.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python reproduce.py --data-dir data --output outputs/final --catboost-device GPU --threads 4
```

`--catboost-device CPU` is available for machines without a compatible GPU. The selected stacking entry used GPU CatBoost, so a CPU retraining may produce different probabilities. The other models run on CPU. Training on the full competition data is substantial and may take hours; the script stores each fold in `outputs/final/cache/` and resumes from completed folds when rerun with unchanged code, inputs, package versions, device, and thread count. The output directory can be changed freely. No machine-specific path is embedded in the code.

The final files are `outputs/final/submission_weighted.csv` and `outputs/final/submission_stacking.csv`. `manifest.json` stores input and source hashes, and `summary.json` stores output hashes and local cross-validation information. The CSVs contain `id` and `Will_Buy_EV` probabilities in the order of `sample_submission.csv`.

## Method

The weighted entry averages rank-normalized LightGBM predictions from seeds 42 and 2026 at weight 0.75, then adds a weight of 0.25 from an XGBoost-CatBoost rank ensemble. The latter uses two XGBoost income target-encoding variants and one CatBoost model. Its test predictions are trained on the full competition training set.

The stacking entry uses five out-of-fold prediction columns: both LightGBM seeds, the XGBoost-CatBoost ensemble, a LightGBM model with grouped interaction constraints, and a CatBoost model over native categorical features. It fits a nonnegative logistic model on rank-transformed predictions in five-fold cross-fitting for each of the two seeds. Fold test scores are averaged, then the two seed scores are rank-averaged. Target encodings are cross-fitted inside each training fold.

The LightGBM feature recipe adapts [Naji's public notebook](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94607-lb-0-94638); its Apache 2.0 license is included in `THIRD_PARTY_LICENSE.txt`. The implementation was rewritten here to make the selected pipeline directly runnable. The original data is credited to [its Kaggle dataset author](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety). The final scores above are historical competition results, not a guarantee that retraining on different hardware will reproduce identical CSV hashes or leaderboard scores.
