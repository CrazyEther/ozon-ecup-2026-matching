# Ozon E-CUP 2026 Matching

This repository is an offline, container-ready solution template for product-pair
matching.  It scores each candidate pair with a locally stored multilingual
cross-encoder and writes the exact `id1,id2,predict` CSV required by the
competition.

## Layout

- `run.py` — competition entry point; it never downloads from the internet.
- `src/inference.py` — memory-conscious batched inference and CSV validation.
- `train.py` — fine-tunes a Hugging Face sequence-classification checkpoint on
  human labels, optionally followed by LLM-labelled pairs.
- `train_lora.py` — memory-efficient LoRA training with automatic multi-GPU DDP.
- `prepare_model.py` — downloads the initial model once during development so it
  can be included in the submission archive.

## Local model

The default model location is `models/product-matcher`.  Train it with
`train.py`, or put a compatible `AutoModelForSequenceClassification` checkpoint
there.  The recommended starting checkpoint is
`microsoft/mdeberta-v3-base`; it is multilingual and handles Russian product
descriptions.  The checkpoint, tokenizer, and model weights must be included in
the final archive/image.

## Prepare and train

```bash
python prepare_model.py --model microsoft/mdeberta-v3-base --output models/product-matcher
python train.py --items-path /data/items_human.parquet --matches-path /data/matches.parquet \
  --model-path models/product-matcher --output-dir models/product-matcher
```

For a second training stage with soft LLM labels, pass `matches_llm.parquet` as
`--matches-path` and use a copy of the human-fine-tuned model as
`--model-path`.  Its 0–1 targets are consumed directly as soft labels.
`--max-train-rows` is useful to establish a fast, reproducible experiment
before training on all data.

### LoRA on one or more GPUs

`train_lora.py` automatically detects all visible CUDA GPUs. With two Kaggle
T4s, the usual `python train_lora.py ...` command relaunches two DDP workers;
no explicit `torchrun` command is required. `--batch-size` remains the global
batch size: `--batch-size 64` on two GPUs assigns 32 examples to each GPU.
The value must be divisible by the GPU count. Pass `--single-gpu` only when
automatic multi-GPU execution must be disabled.

```bash
python train_lora.py \
  --items-path /kaggle/working/data/items_human.parquet \
  --matches-path /kaggle/working/data/matches.parquet \
  --model-path /kaggle/working/models/gte \
  --trust-remote-code \
  --output-dir /kaggle/working/gte-lora-r16 \
  --epochs 1 --batch-size 64 --max-length 128 \
  --learning-rate 2e-4 --lora-rank 16 --lora-alpha 32 \
  --checkpoint-steps 250 --keep-checkpoints 2
```

Only rank zero writes checkpoints and the best adapter. Validation is sharded
without padding, then gathered before macro PR-AUC is calculated, so multi-GPU
validation does not duplicate rows.

## Test the competition interface

```bash
python -u run.py -i /data/items_human.parquet -m /data/matches.parquet -o submit.csv
```

The entry point supports both `--output_path` (official baseline spelling) and
`--output-path`; similarly it accepts `-i/--items_path` and
`-m/--matches_path`.

## Package

Use `metadata.json` unchanged only if the base image has all listed packages.
Before submitting, ensure `models/product-matcher` is present and run the smoke
test above in an offline container. `Dockerfile` builds a self-contained image;
after pushing it to Docker Hub, replace the `image` value in `metadata.json`
with its immutable image tag.

