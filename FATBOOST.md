# FatBoost: expanded CatBoost matcher

`train_catboost.py` now trains a substantially richer product matcher. It
combines 46 deterministic/TF-IDF numerical features, four categorical fields,
three native CatBoost text fields, and optional tiny-transformer similarities.

## Signals

- RapidFuzz ratio, WRatio, partial ratio, token-sort and token-set scores;
- word 1-2 gram and character 3-5 gram TF-IDF cosine similarities;
- model/SKU codes with Cyrillic/Latin confusable normalization;
- normalized mass, volume, length, power, storage, frequency and diagonals;
- package quantity, brand and per-attribute match/conflict signals;
- native CatBoost text over pair names, shared tokens and differing tokens;
- optional five distances from a tiny locally stored Transformer embedding.

The fitted TF-IDF vocabularies are stored in `*.features.joblib`. Keep that
file beside the `*.cbm` and `*.json`; it is required for inference.

## Colab smoke test

```bash
pip install -r requirements-fatboost.txt

python train_catboost.py \
  --items-path /content/data/items_human.parquet \
  --matches-path /content/data/matches.parquet \
  --output /content/drive/MyDrive/ozon-matching/fatboost-smoke.cbm \
  --mode weighted-classifier \
  --max-rows 5000 --iterations 300 --depth 7
```

This saves three files: `.cbm`, `.json`, and `.features.joblib`.

## Full experiments

Run the category-balanced classifier first:

```bash
python train_catboost.py \
  --items-path /content/data/items_human.parquet \
  --matches-path /content/data/matches.parquet \
  --output /content/drive/MyDrive/ozon-matching/fatboost-weighted.cbm \
  --mode weighted-classifier \
  --iterations 2000 --depth 8 --learning-rate 0.03
```

Then compare it with direct category-MAP ranking:

```bash
python train_catboost.py \
  --items-path /content/data/items_human.parquet \
  --matches-path /content/data/matches.parquet \
  --output /content/drive/MyDrive/ozon-matching/fatboost-ranker.cbm \
  --mode ranker \
  --iterations 2000 --depth 8 --learning-rate 0.03
```

`ranker` sorts rows by category and uses each category as one CatBoost group
with `YetiRank:mode=MAP`. Both runs print the exact external macro PR-AUC.

## Small edge embedding profile

Embeddings are optional because TF-IDF + CatBoost is substantially faster on
CPU. To test semantic features, `cointegrated/rubert-tiny` is a 12M-parameter,
roughly 45 MB Russian/English encoder and needs no `sentence-transformers`
package. It is smaller than `rubert-tiny2` (29.4M parameters). Download it once
during training and save an offline copy:

```bash
python train_catboost.py \
  --items-path /content/data/items_human.parquet \
  --matches-path /content/data/matches.parquet \
  --output /content/drive/MyDrive/ozon-matching/fatboost-tiny-embed.cbm \
  --mode weighted-classifier \
  --embedding-model cointegrated/rubert-tiny \
  --embedding-allow-download \
  --embedding-copy-to /content/drive/MyDrive/ozon-matching/rubert-tiny \
  --embedding-batch-size 512 --embedding-max-length 96
```

For a final offline package, copy the encoder directory into the package and
override its location with `--embedding-model` during prediction.

## Offline prediction

```bash
python predict_catboost.py \
  --items-path /data/items_human.parquet \
  --matches-path /data/matches.parquet \
  --model /models/fatboost-weighted.cbm \
  --output /output/predictions.csv
```

The model type is read from the adjacent `.json` manifest. If the manifest is
not available, pass `--model-type classifier` or `--model-type ranker`.

For the embedding build, also pass the local encoder directory:

```bash
  --embedding-model /models/rubert-tiny
```

## Speed profiles

- Fastest edge: add `--disable-native-text`, do not use `--embedding-model`.
- Balanced: native text + TF-IDF, no neural embedding (default).
- FatBoost: native text + TF-IDF + tiny embedding.

Always compare macro PR-AUC and measured inference time. A tiny embedding may
improve score but is not automatically worth its CPU cost.
