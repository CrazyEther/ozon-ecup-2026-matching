"""Recreate a clean holdout unseen by both the LoRA and FatBoost models."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def fatboost_training_source_rows(
    matches: pd.DataFrame,
    categories: pd.Series,
    *,
    max_rows: int,
    validation_fraction: float,
    seed: int,
) -> set[int]:
    sample = matches.copy()
    sample["_source_row"] = sample.index
    sample = sample.sample(n=min(max_rows, len(sample)), random_state=seed).reset_index(drop=True)
    sample["category"] = sample.id1.map(categories).fillna("__missing__").astype(str)
    rng = np.random.default_rng(seed)
    validation: list[int] = []
    for _, group in sample.groupby(["category", "target"], dropna=False, sort=False):
        count = max(1, int(round(len(group) * validation_fraction))) if len(group) > 1 else 0
        if count:
            validation.extend(
                rng.choice(group.index.to_numpy(), size=min(count, len(group) - 1), replace=False)
            )
    train_indices = sample.index.difference(pd.Index(sorted(validation)))
    return set(sample.loc[train_indices, "_source_row"].astype(int))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--matches-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lora-validation-fraction", type=float, default=0.1)
    parser.add_argument("--fatboost-validation-fraction", type=float, default=0.1)
    parser.add_argument("--fatboost-max-rows", type=int, default=5000)
    args = parser.parse_args()

    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2", "target"])
    items = pd.read_parquet(args.items_path, columns=["id", "category"])
    categories = items.drop_duplicates("id").set_index("id").category
    category1 = matches.id1.map(categories)
    if category1.isna().any():
        raise ValueError("Some id1 values are absent from items parquet")

    # Exact split used by train.py/train_lora.py with the default seed.
    split_frame = matches.assign(category1=category1.astype(str))
    lora_validation = split_frame.groupby("category1", group_keys=False).sample(
        frac=args.lora_validation_fraction,
        random_state=args.seed,
    )

    # The smoke FatBoost used a different split. Remove its supervised training
    # rows from LoRA validation, leaving examples unseen by either model.
    fatboost_train_rows = fatboost_training_source_rows(
        matches,
        categories,
        max_rows=args.fatboost_max_rows,
        validation_fraction=args.fatboost_validation_fraction,
        seed=args.seed,
    )
    clean_indices = [index for index in lora_validation.index if index not in fatboost_train_rows]
    holdout = matches.loc[clean_indices, ["id1", "id2", "target"]].reset_index(drop=True)
    if not len(holdout):
        raise RuntimeError("Clean ensemble holdout is empty")
    holdout.to_parquet(args.output, index=False)
    print(
        f"Saved {len(holdout):,} clean holdout pairs to {args.output}; "
        f"removed {len(lora_validation) - len(holdout):,} FatBoost training rows",
    )


if __name__ == "__main__":
    main()
