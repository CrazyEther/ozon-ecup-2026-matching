"""Fast, offline, order-preserving inference for candidate product pairs."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .text import add_product_text


def _load_pairs(items_path: str | Path, matches_path: str | Path) -> pd.DataFrame:
    matches = pd.read_parquet(matches_path)
    missing = {"id1", "id2"}.difference(matches.columns)
    if missing:
        raise ValueError(f"Matches parquet is missing columns: {sorted(missing)}")
    pairs = matches.loc[:, ["id1", "id2"]].copy()
    pairs["_row"] = np.arange(len(pairs), dtype=np.int64)

    items = add_product_text(pd.read_parquet(items_path))
    if items.id.duplicated().any():
        raise ValueError("Items parquet has duplicate product IDs")
    left = items.rename(columns={"id": "id1", "text": "text1", "category": "category1"})
    right = items.rename(columns={"id": "id2", "text": "text2", "category": "category2"})
    pairs = pairs.merge(left, on="id1", how="left", validate="many_to_one")
    pairs = pairs.merge(right, on="id2", how="left", validate="many_to_one")
    return pairs


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits[:, 0])
    return torch.softmax(logits, dim=-1)[:, 1]


def predict_to_csv(
    *, items_path: str | Path, matches_path: str | Path, output_path: str | Path,
    model_path: str | Path, batch_size: int, max_length: int, trust_remote_code: bool = False,
    adapter_path: str | Path | None = None,
) -> None:
    pairs = _load_pairs(items_path, matches_path)
    result = pairs.loc[:, ["id1", "id2"]].copy()
    scores = np.zeros(len(pairs), dtype=np.float32)
    valid = pairs.text1.notna() & pairs.text2.notna()
    if not valid.any():
        raise ValueError("None of the supplied pair IDs were found in items parquet")

    device = _device()
    print(f"Loaded {len(pairs):,} pairs; scoring {valid.sum():,} on {device}.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True, trust_remote_code=trust_remote_code,
    )
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=trust_remote_code,
        torch_dtype=amp_dtype if device.type == "cuda" else None,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model = model.to(device).eval()

    valid_rows = np.flatnonzero(valid.to_numpy())
    with torch.inference_mode():
        for start in tqdm(range(0, len(valid_rows), batch_size), desc="Scoring", unit="batch"):
            rows = valid_rows[start : start + batch_size]
            batch = pairs.iloc[rows]
            encoded = tokenizer(
                batch.text1.tolist(), batch.text2.tolist(), padding=True,
                truncation=True, max_length=max_length, return_tensors="pt",
            )
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=amp_dtype):
                    batch_scores = _logits_to_scores(model(**encoded).logits)
            else:
                batch_scores = _logits_to_scores(model(**encoded).logits)
            scores[rows] = batch_scores.float().cpu().numpy()

    result["predict"] = scores
    if len(result) != len(pairs) or result[["id1", "id2"]].isna().any().any():
        raise RuntimeError("Output validation failed: pair IDs were not preserved")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result):,} predictions to {output_path}")

