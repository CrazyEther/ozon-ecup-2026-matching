"""Train and validate the small lexical companion model on human labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.lexical import FEATURE_NAMES, pair_features


def macro_ap(frame: pd.DataFrame, score) -> float:
    values = [average_precision_score(group.target, score[group.index])
              for _, group in frame.groupby("category") if group.target.nunique() == 2]
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--matches-path", required=True)
    parser.add_argument("--output", default="artifacts/lexical.joblib")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    items = pd.read_parquet(args.items_path)
    matches = pd.read_parquet(args.matches_path)
    if "target" not in matches:
        raise ValueError("Training matches must include target")
    categories = items.set_index("id").category
    frame = matches.copy()
    frame["category"] = frame.id1.map(categories)
    features = pair_features(items, frame)
    validation = frame.groupby("category", group_keys=False).sample(frac=0.1, random_state=args.seed).index
    train_mask = ~frame.index.isin(validation)
    model = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=args.seed)
    model.fit(features[train_mask], frame.loc[train_mask, "target"])
    scores = model.predict_proba(features)[:, 1]
    print(f"Validation macro PR-AUC: {macro_ap(frame.loc[validation], scores):.6f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, args.output)
    print(f"Saved lexical model to {args.output}")


if __name__ == "__main__":
    main()

