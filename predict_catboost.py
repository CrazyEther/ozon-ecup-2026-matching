"""Offline inference for a fitted FatBoost model and feature builder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from src.fatboost import FatBoostFeatureBuilder, TinyTextEncoder, sigmoid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--matches-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output", default="fatboost_predictions.csv")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embedding-max-length", type=int, default=96)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--model-type", choices=["auto", "classifier", "ranker"], default="auto")
    args = parser.parse_args()

    model_path = Path(args.model)
    feature_path = Path(args.features) if args.features else model_path.with_suffix(".features.joblib")
    builder: FatBoostFeatureBuilder = joblib.load(feature_path)
    encoder = None
    if builder.embedding_model:
        encoder_path = args.embedding_model or builder.embedding_model
        encoder = TinyTextEncoder(
            encoder_path,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            device=args.embedding_device,
            local_files_only=True,
            pooling=builder.embedding_pooling,
        )

    items = pd.read_parquet(args.items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    involved_ids = pd.unique(pd.concat([matches.id1, matches.id2], ignore_index=True))
    items = items.loc[items.id.isin(involved_ids)].reset_index(drop=True)
    missing_items = len(involved_ids) - items.id.nunique()
    if missing_items:
        raise ValueError(f"items parquet is missing {missing_items} product ids used by matches")
    print(f"Using {len(items):,} referenced products for {len(matches):,} pairs", flush=True)
    features = builder.transform(items, matches, encoder=encoder)
    pool = Pool(
        features,
        cat_features=builder.cat_feature_names,
        text_features=builder.text_feature_names,
    )
    model_type = args.model_type
    metadata_path = model_path.with_suffix(".json")
    if model_type == "auto" and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_type = "ranker" if metadata.get("mode") == "ranker" else "classifier"
    if model_type == "auto":
        raise ValueError("Cannot infer model type: keep the .json manifest or pass --model-type")
    if model_type == "ranker":
        model: CatBoostClassifier | CatBoostRanker = CatBoostRanker()
        model.load_model(model_path)
        scores = sigmoid(np.asarray(model.predict(pool)))
    else:
        model = CatBoostClassifier()
        model.load_model(model_path)
        scores = np.asarray(model.predict_proba(pool))[:, 1]
    output = matches.loc[:, ["id1", "id2"]].copy()
    output["predict"] = np.clip(scores, 0.0, 1.0)
    if len(output) != len(matches) or output.predict.isna().any():
        raise RuntimeError("Invalid prediction output")
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
