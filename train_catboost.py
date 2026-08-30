"""Train a CPU-only CatBoost product matcher on lexical and attribute features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from src.lexical import FEATURE_NAMES, pair_features


def macro_pr_auc(categories: pd.Series, targets: pd.Series, scores: np.ndarray) -> float:
    frame = pd.DataFrame(
        {
            "category": categories.reset_index(drop=True),
            "target": targets.reset_index(drop=True),
            "score": scores,
        }
    )
    values = [
        average_precision_score(group.target, group.score)
        for _, group in frame.groupby("category", dropna=False)
        if group.target.nunique() == 2
    ]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--matches-path", required=True)
    parser.add_argument("--output", default="artifacts/product-matcher.cbm")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--thread-count", type=int, default=-1)
    args = parser.parse_args()

    items = pd.read_parquet(args.items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2", "target"])
    if matches.target.isna().any() or not matches.target.between(0, 1).all():
        raise ValueError("target must be in [0, 1]")

    categories = items.set_index("id").category
    frame = matches.copy()
    frame["category"] = frame.id1.map(categories).fillna("__missing__").astype(str)
    print(f"Building {len(FEATURE_NAMES)} lexical features for {len(frame):,} pairs...", flush=True)
    feature_frame = pd.DataFrame(pair_features(items, frame), columns=FEATURE_NAMES)
    feature_frame["category"] = frame.category.to_numpy()

    validation_index = frame.groupby("category", group_keys=False).sample(
        frac=args.validation_fraction, random_state=args.seed,
    ).index
    train_mask = ~frame.index.isin(validation_index)
    train_features = feature_frame.loc[train_mask]
    valid_features = feature_frame.loc[validation_index]
    train_target = frame.loc[train_mask, "target"]
    valid_target = frame.loc[validation_index, "target"]
    print(
        f"Training pairs: {len(train_features):,}; validation pairs: {len(valid_features):,}",
        flush=True,
    )

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        l2_leaf_reg=5.0,
        random_seed=args.seed,
        thread_count=args.thread_count,
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(
        train_features,
        train_target,
        cat_features=["category"],
        eval_set=(valid_features, valid_target),
        early_stopping_rounds=args.early_stopping_rounds,
        use_best_model=True,
    )
    valid_scores = model.predict_proba(valid_features)[:, 1]
    score = macro_pr_auc(
        frame.loc[validation_index, "category"], valid_target, valid_scores,
    )
    print(f"Validation macro PR-AUC: {score:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output)
    metadata = {
        "type": "catboost_lexical",
        "validation_macro_pr_auc": score,
        "best_iteration": model.get_best_iteration(),
        "feature_names": [*FEATURE_NAMES, "category"],
        "seed": args.seed,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Saved CatBoost model to {output}")


if __name__ == "__main__":
    main()

