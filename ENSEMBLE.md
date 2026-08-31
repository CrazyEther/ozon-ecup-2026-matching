# LoRA + FatBoost ensemble

The ensemble keeps the existing neural matcher intact and adds the saved
FatBoost model as a second, complementary scorer. No retraining is required.

## Required offline artifacts

Place the files in the final solution directory as follows:

```text
models/product-matcher/                 # existing merged LoRA checkpoint
models/fatboost/product-matcher.cbm
models/fatboost/product-matcher.json
models/fatboost/product-matcher.features.joblib
```

If LoRA was not merged, keep the base checkpoint in `product-matcher` and put
the adapter in `models/lora`; add `--adapter-path models/lora` to every command.

## Select the blend on a clean holdout

The saved 5k FatBoost smoke model learned on 4,501 rows. This command
reconstructs LoRA validation and removes those FatBoost training rows:

```bash
python prepare_ensemble_holdout.py \
  --items-path /kaggle/working/data/items_human.parquet \
  --matches-path /kaggle/working/data/matches.parquet \
  --output /kaggle/working/ensemble-holdout.parquet \
  --fatboost-max-rows 5000
```

Generate the two component predictions without training:

```bash
python -u run.py \
  -i /kaggle/working/data/items_human.parquet \
  -m /kaggle/working/ensemble-holdout.parquet \
  -o /kaggle/working/lora-holdout.csv \
  --model-path /kaggle/working/models/product-matcher \
  --batch-size 384 --max-length 128

python -u predict_catboost.py \
  --items-path /kaggle/working/data/items_human.parquet \
  --matches-path /kaggle/working/ensemble-holdout.parquet \
  --model /kaggle/working/models/fatboost/product-matcher.cbm \
  --output /kaggle/working/fatboost-holdout.csv

python tune_ensemble.py \
  --items-path /kaggle/working/data/items_human.parquet \
  --matches-path /kaggle/working/ensemble-holdout.parquet \
  --lora-predictions /kaggle/working/lora-holdout.csv \
  --fatboost-predictions /kaggle/working/fatboost-holdout.csv
```

The last line prints the best `--ensemble-method` and `--fatboost-weight`.
Rank blending is the provisional default because the two models have very
different probability calibration.

## Final inference

Replace the example weight below with the measured one:

```bash
python -u run.py \
  -i items.parquet -m matches.parquet -o predictions.csv \
  --model-path models/product-matcher \
  --fatboost-model models/fatboost/product-matcher.cbm \
  --ensemble-method rank --fatboost-weight 0.50 \
  --batch-size 384 --max-length 128
```

For the competition, put the same flags in `metadata.json`. The runtime image
must contain CatBoost, joblib, RapidFuzz and scikit-learn; `Dockerfile.runtime`
now installs pinned versions of them. Run the 115k and 275k timing checks before
uploading because total time is neural inference plus FatBoost feature building.
