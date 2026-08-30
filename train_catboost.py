"""Train the FatBoost product matcher.

The default weighted-classifier mode is the safest CatBoost baseline. The
ranker mode uses category as group_id and directly optimizes mean AP across
categories with ``YetiRank:mode=MAP``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.metrics import average_precision_score

from src.fatboost import FatBoostFeatureBuilder, TinyTextEncoder, sigmoid


def macro_pr_auc(categories: pd.Series, targets: pd.Series, scores: np.ndarray) -> float:
    frame = pd.DataFrame({
        "category": categories.reset_index(drop=True),
        "target": targets.reset_index(drop=True),
        "score": np.asarray(scores),
    })
    values = [
        average_precision_score(group.target, group.score)
        for _, group in frame.groupby("category", dropna=False)
        if group.target.nunique() == 2
    ]
    return float(np.mean(values)) if values else float("nan")


def split_indices(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.Index, pd.Index]:
    """Deterministic category-and-label stratified row split."""
    rng = np.random.default_rng(seed)
    validation: list[int] = []
    for _, group in frame.groupby(["category", "target"], dropna=False, sort=False):
        count = max(1, int(round(len(group) * fraction))) if len(group) > 1 else 0
        if count:
            validation.extend(
                rng.choice(group.index.to_numpy(), size=min(count, len(group) - 1), replace=False)
            )
    validation_index = pd.Index(sorted(validation))
    return frame.index.difference(validation_index), validation_index


def balanced_weights(frame: pd.DataFrame, indices: pd.Index) -> np.ndarray:
    """Give every (category, target) stratum the same total training weight."""
    subset = frame.loc[indices, ["category", "target"]]
    counts = subset.groupby(["category", "target"], dropna=False).size()
    weights = np.asarray([
        1.0 / counts.loc[(category, target)]
        for category, target in subset.itertuples(index=False)
    ], dtype=np.float64)
    return (weights / weights.mean()).astype(np.float32)


def make_pool(
    features: pd.DataFrame,
    frame: pd.DataFrame,
    indices: pd.Index,
    builder: FatBoostFeatureBuilder,
    ranking: bool,
    weights: np.ndarray | None = None,
) -> tuple[Pool, pd.Index]:
    selected = indices
    if ranking:
        # CatBoost requires rows of one group to be contiguous.
        selected = frame.loc[indices].sort_values("category", kind="stable").index
    data = features.loc[selected].reset_index(drop=True)
    labels = frame.loc[selected, "target"].to_numpy(dtype=np.float32)
    kwargs: dict[str, object] = {
        "data": data,
        "label": labels,
        "cat_features": builder.cat_feature_names,
        "text_features": builder.text_feature_names,
    }
    if ranking:
        kwargs["group_id"] = frame.loc[selected, "category"].astype(str).to_numpy()
    elif weights is not None:
        weight_by_index = pd.Series(weights, index=indices)
        kwargs["weight"] = weight_by_index.loc[selected].to_numpy()
    return Pool(**kwargs), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--matches-path", required=True)
    parser.add_argument("--output", default="artifacts/fatboost/product-matcher.cbm")
    parser.add_argument(
        "--mode", choices=["classifier", "weighted-classifier", "ranker"],
        default="weighted-classifier",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--random-strength", type=float, default=0.5)
    parser.add_argument("--bagging-temperature", type=float, default=0.5)
    parser.add_argument("--early-stopping-rounds", type=int, default=180)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--word-max-features", type=int, default=120_000)
    parser.add_argument("--char-max-features", type=int, default=160_000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--disable-native-text", action="store_true")
    parser.add_argument(
        "--embedding-model", default=None,
        help="Optional local path or HF id. cointegrated/rubert-tiny is the smallest recommended edge encoder.",
    )
    parser.add_argument("--embedding-allow-download", action="store_true")
    parser.add_argument("--embedding-copy-to", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embedding-max-length", type=int, default=96)
    parser.add_argument("--embedding-pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    items = pd.read_parquet(args.items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2", "target"])
    if args.max_rows is not None:
        matches = matches.sample(
            n=min(args.max_rows, len(matches)), random_state=args.seed,
        ).reset_index(drop=True)
    if matches.target.isna().any() or not matches.target.between(0, 1).all():
        raise ValueError("target must be in [0, 1]")

    # A smoke run must not parse/vectorize every row of items_human. Keep only
    # products referenced by the sampled pairs; the competition inference input
    # follows the same contract and contains only participating products.
    involved_ids = pd.unique(pd.concat([matches.id1, matches.id2], ignore_index=True))
    items = items.loc[items.id.isin(involved_ids)].reset_index(drop=True)
    missing_items = len(involved_ids) - items.id.nunique()
    if missing_items:
        raise ValueError(f"items parquet is missing {missing_items} product ids used by matches")
    print(f"Using {len(items):,} referenced products for {len(matches):,} pairs", flush=True)

    item_categories = items.set_index("id").category
    frame = matches.copy()
    frame["category"] = frame.id1.map(item_categories).fillna("__missing__").astype(str)
    train_index, validation_index = split_indices(frame, args.validation_fraction, args.seed)
    print(
        f"Pairs: {len(frame):,}; training: {len(train_index):,}; validation: {len(validation_index):,}",
        flush=True,
    )

    encoder = None
    embedding_model = args.embedding_model
    if embedding_model:
        print(f"Loading optional tiny encoder: {embedding_model}", flush=True)
        encoder = TinyTextEncoder(
            embedding_model,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            device=args.embedding_device,
            local_files_only=not args.embedding_allow_download,
            pooling=args.embedding_pooling,
        )
        if args.embedding_copy_to:
            encoder.save_pretrained(args.embedding_copy_to)
            embedding_model = str(Path(args.embedding_copy_to))

    builder = FatBoostFeatureBuilder(
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        min_df=args.tfidf_min_df,
        native_text=not args.disable_native_text,
        embedding_model=embedding_model,
        embedding_pooling=args.embedding_pooling,
    )
    print(
        "Building FatBoost features: model/SKU codes, normalized units, fuzzy metrics, "
        "TF-IDF, native CatBoost text" + (", tiny embeddings" if encoder else "") + "...",
        flush=True,
    )
    features = builder.fit_transform(items, frame, encoder=encoder)
    ranking = args.mode == "ranker"
    train_weights = balanced_weights(frame, train_index) if args.mode == "weighted-classifier" else None
    valid_weights = balanced_weights(frame, validation_index) if args.mode == "weighted-classifier" else None
    train_pool, _ = make_pool(features, frame, train_index, builder, ranking, train_weights)
    valid_pool, valid_order = make_pool(
        features, frame, validation_index, builder, ranking, valid_weights,
    )

    common = dict(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        bagging_temperature=args.bagging_temperature,
        random_seed=args.seed,
        thread_count=args.thread_count,
        task_type="CPU",  # YetiRank MAP and native text are CPU-oriented.
        allow_writing_files=False,
        verbose=50,
    )
    if ranking:
        model: CatBoostClassifier | CatBoostRanker = CatBoostRanker(
            loss_function="YetiRank:mode=MAP",
            eval_metric="MAP",
            **common,
        )
    else:
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC:type=Classic",
            **common,
        )
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=args.early_stopping_rounds,
        use_best_model=True,
    )
    if ranking:
        valid_scores = sigmoid(np.asarray(model.predict(valid_pool)))
    else:
        valid_scores = np.asarray(model.predict_proba(valid_pool))[:, 1]
    score = macro_pr_auc(
        frame.loc[valid_order, "category"], frame.loc[valid_order, "target"], valid_scores,
    )
    print(f"Validation macro PR-AUC: {score:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output)
    builder_output = output.with_suffix(".features.joblib")
    joblib.dump(builder, builder_output, compress=3)
    metadata = {
        "type": "fatboost_product_matcher",
        "mode": args.mode,
        "validation_macro_pr_auc": score,
        "best_iteration": model.get_best_iteration(),
        "feature_count": len(features.columns),
        "numeric_features": builder.numeric_feature_names,
        "categorical_features": builder.cat_feature_names,
        "text_features": builder.text_feature_names,
        "embedding_model": builder.embedding_model,
        "embedding_pooling": builder.embedding_pooling,
        "model_file": output.name,
        "feature_builder_file": builder_output.name,
        "seed": args.seed,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Saved model to {output}")
    print(f"Saved fitted feature builder to {builder_output}")


if __name__ == "__main__":
    main()
