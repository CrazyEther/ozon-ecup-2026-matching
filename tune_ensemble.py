"""Select LoRA/FatBoost blend method and weight on a labelled holdout."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.ensemble import blend_scores


def macro_pr_auc(targets: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> float:
    values = []
    frame = pd.DataFrame({"target": targets, "score": scores, "group": groups})
    for _, part in frame.groupby("group", sort=False):
        if part.target.nunique() == 2:
            values.append(average_precision_score(part.target, part.score))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-path", required=True, help="Labelled holdout parquet")
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--lora-predictions", required=True)
    parser.add_argument("--fatboost-predictions", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    labels = pd.read_parquet(args.matches_path, columns=["id1", "id2", "target"])
    lora = pd.read_csv(args.lora_predictions)
    fatboost = pd.read_csv(args.fatboost_predictions)
    for name, predictions in (("LoRA", lora), ("FatBoost", fatboost)):
        if not predictions[["id1", "id2"]].equals(labels[["id1", "id2"]]):
            raise ValueError(f"{name} predictions do not match holdout IDs/order")

    items = pd.read_parquet(args.items_path, columns=["id", "category"])
    category_map = items.drop_duplicates("id").set_index("id").category
    groups = labels.id1.map(category_map).fillna(labels.id2.map(category_map)).fillna("__missing__")
    targets = labels.target.to_numpy()
    candidates = []
    weights = np.arange(0.0, 1.0 + args.step / 2.0, args.step)
    for method in ("rank", "logit", "linear"):
        for weight in weights:
            scores = blend_scores(
                lora.predict.to_numpy(),
                fatboost.predict.to_numpy(),
                fatboost_weight=float(weight),
                method=method,
                groups=groups.to_numpy(),
            )
            candidates.append((macro_pr_auc(targets, scores, groups.to_numpy()), method, weight))
    candidates.sort(reverse=True)
    print("Top ensemble settings:")
    for score, method, weight in candidates[:10]:
        print(f"macro PR-AUC={score:.6f}  method={method:6s}  FatBoost weight={weight:.2f}")
    best_score, best_method, best_weight = candidates[0]
    print(
        f"\nUse: --ensemble-method {best_method} --fatboost-weight {best_weight:.2f} "
        f"(holdout macro PR-AUC {best_score:.6f})"
    )


if __name__ == "__main__":
    main()
