"""Reusable, offline FatBoost inference used by the ensemble entry point."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from .fatboost import FatBoostFeatureBuilder, TinyTextEncoder, sigmoid


def predict_fatboost_scores(
    *,
    items_path: str | Path,
    matches_path: str | Path,
    model_path: str | Path,
    feature_path: str | Path | None = None,
    model_type: str = "auto",
    embedding_model: str | Path | None = None,
    embedding_batch_size: int = 256,
    embedding_max_length: int = 96,
    embedding_device: str | None = None,
) -> pd.DataFrame:
    """Return order-preserving ``id1,id2,predict`` FatBoost scores."""
    model_path = Path(model_path)
    feature_path = (
        Path(feature_path) if feature_path else model_path.with_suffix(".features.joblib")
    )
    builder: FatBoostFeatureBuilder = joblib.load(feature_path)
    encoder = None
    if builder.embedding_model:
        encoder = TinyTextEncoder(
            embedding_model or builder.embedding_model,
            batch_size=embedding_batch_size,
            max_length=embedding_max_length,
            device=embedding_device,
            local_files_only=True,
            pooling=builder.embedding_pooling,
        )

    items = pd.read_parquet(
        items_path, columns=["id", "name", "attributes", "category"],
    )
    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    involved_ids = pd.unique(pd.concat([matches.id1, matches.id2], ignore_index=True))
    items = items.loc[items.id.isin(involved_ids)].reset_index(drop=True)
    missing_items = len(involved_ids) - items.id.nunique()
    if missing_items:
        raise ValueError(f"items parquet is missing {missing_items} product ids used by matches")
    print(f"FatBoost: using {len(items):,} products for {len(matches):,} pairs", flush=True)

    features = builder.transform(items, matches, encoder=encoder)
    pool = Pool(
        features,
        cat_features=builder.cat_feature_names,
        text_features=builder.text_feature_names,
    )
    metadata_path = model_path.with_suffix(".json")
    if model_type == "auto" and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_type = "ranker" if metadata.get("mode") == "ranker" else "classifier"
    if model_type == "auto":
        raise ValueError("Cannot infer FatBoost model type: keep its .json or specify it")
    if model_type == "ranker":
        model: CatBoostClassifier | CatBoostRanker = CatBoostRanker()
        model.load_model(model_path)
        scores = sigmoid(np.asarray(model.predict(pool)))
    elif model_type == "classifier":
        model = CatBoostClassifier()
        model.load_model(model_path)
        scores = np.asarray(model.predict_proba(pool))[:, 1]
    else:
        raise ValueError(f"Unknown FatBoost model type: {model_type}")

    result = matches.loc[:, ["id1", "id2"]].copy()
    result["predict"] = np.clip(scores, 0.0, 1.0)
    if len(result) != len(matches) or result.predict.isna().any():
        raise RuntimeError("Invalid FatBoost prediction output")
    return result
