"""LoRA + FatBoost ensemble inference and calibration-independent blending."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd


def _validate_scores(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} scores must be a finite one-dimensional array")
    return np.clip(values, 0.0, 1.0)


def blend_scores(
    lora_scores: np.ndarray,
    fatboost_scores: np.ndarray,
    *,
    fatboost_weight: float,
    method: str = "rank",
    groups: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Blend two models; rank blending is robust to unlike calibration."""
    if not 0.0 <= fatboost_weight <= 1.0:
        raise ValueError("fatboost_weight must be between 0 and 1")
    lora = _validate_scores(lora_scores, "LoRA")
    fatboost = _validate_scores(fatboost_scores, "FatBoost")
    if lora.shape != fatboost.shape:
        raise ValueError("LoRA and FatBoost prediction lengths differ")

    if method == "linear":
        result = (1.0 - fatboost_weight) * lora + fatboost_weight * fatboost
    elif method == "logit":
        epsilon = 1e-6
        lora_logit = np.log(np.clip(lora, epsilon, 1 - epsilon) / np.clip(1 - lora, epsilon, 1))
        fatboost_logit = np.log(
            np.clip(fatboost, epsilon, 1 - epsilon) / np.clip(1 - fatboost, epsilon, 1)
        )
        mixed = (1.0 - fatboost_weight) * lora_logit + fatboost_weight * fatboost_logit
        result = 1.0 / (1.0 + np.exp(-np.clip(mixed, -30.0, 30.0)))
    elif method == "rank":
        if groups is None:
            groups = np.zeros(len(lora), dtype=np.int8)
        group_series = pd.Series(np.asarray(groups), copy=False)
        lora_rank = pd.Series(lora).groupby(group_series, sort=False).rank(method="average", pct=True)
        fatboost_rank = pd.Series(fatboost).groupby(group_series, sort=False).rank(
            method="average", pct=True,
        )
        result = (1.0 - fatboost_weight) * lora_rank.to_numpy() + fatboost_weight * fatboost_rank.to_numpy()
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _pair_categories(items_path: str | Path, pairs: pd.DataFrame) -> np.ndarray:
    items = pd.read_parquet(items_path, columns=["id", "category"])
    if items.id.duplicated().any():
        raise ValueError("Items parquet has duplicate product IDs")
    categories = items.set_index("id").category
    left = pairs.id1.map(categories)
    right = pairs.id2.map(categories)
    return left.fillna(right).fillna("__missing_category__").astype(str).to_numpy()


def predict_ensemble_to_csv(
    *,
    items_path: str | Path,
    matches_path: str | Path,
    output_path: str | Path,
    lora_model_path: str | Path,
    lora_adapter_path: str | Path | None,
    fatboost_model_path: str | Path,
    fatboost_feature_path: str | Path | None,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    fatboost_weight: float,
    method: str,
    fatboost_model_type: str = "auto",
) -> None:
    # Keep the cheap blend utilities importable without loading the neural and
    # CatBoost runtimes. The competition entry point imports them only here.
    import torch

    from .fatboost_inference import predict_fatboost_scores
    from .inference import predict_scores as predict_lora_scores

    lora = predict_lora_scores(
        items_path=items_path,
        matches_path=matches_path,
        model_path=lora_model_path,
        adapter_path=lora_adapter_path,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fatboost = predict_fatboost_scores(
        items_path=items_path,
        matches_path=matches_path,
        model_path=fatboost_model_path,
        feature_path=fatboost_feature_path,
        model_type=fatboost_model_type,
    )
    if not lora[["id1", "id2"]].equals(fatboost[["id1", "id2"]]):
        raise RuntimeError("LoRA and FatBoost changed pair order or IDs")
    groups = _pair_categories(items_path, lora) if method == "rank" else None
    result = lora.loc[:, ["id1", "id2"]].copy()
    result["predict"] = blend_scores(
        lora.predict.to_numpy(),
        fatboost.predict.to_numpy(),
        fatboost_weight=fatboost_weight,
        method=method,
        groups=groups,
    )
    if result.predict.isna().any() or not result.predict.between(0, 1).all():
        raise RuntimeError("Invalid ensemble prediction output")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(
        f"Saved {len(result):,} ensemble predictions to {output_path} "
        f"(method={method}, FatBoost weight={fatboost_weight:.3f})",
    )
