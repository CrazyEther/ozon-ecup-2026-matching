"""Fine-tune a local cross-encoder for Ozon product-pair matching.

Human labels are binary; LLM labels can be used as soft targets with exactly the
same loss.  The script deliberately uses only local parquet files and a local
base checkpoint after the initial `prepare_model.py` step.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from src.text import add_product_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--matches-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Required by Alibaba GTE and safe only for a reviewed local checkpoint")
    return parser.parse_args()


def make_pairs(items_path: str, matches_path: str, limit: int, seed: int) -> pd.DataFrame:
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    if matches.target.isna().any() or not matches.target.between(0, 1).all():
        raise ValueError("target must be a probability in [0, 1]")
    if limit and len(matches) > limit:
        matches = matches.sample(limit, random_state=seed)
    items = add_product_text(pd.read_parquet(items_path))
    if items.id.duplicated().any():
        raise ValueError("duplicate IDs in items parquet")
    left = items.rename(columns={"id": "id1", "text": "text1", "category": "category1"})
    right = items.rename(columns={"id": "id2", "text": "text2"})
    pairs = matches.merge(left, on="id1", how="inner", validate="many_to_one")
    pairs = pairs.merge(right, on="id2", how="inner", validate="many_to_one")
    if len(pairs) != len(matches):
        print(f"Warning: dropped {len(matches) - len(pairs):,} pairs without both product cards")
    return pairs.loc[:, ["text1", "text2", "category1", "target"]].reset_index(drop=True)


def split_per_category(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.groupby("category1", group_keys=False).sample(frac=fraction, random_state=seed)
    train = frame.drop(valid.index)
    return train.reset_index(drop=True), valid.reset_index(drop=True)


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.left = frame.text1.tolist()
        self.right = frame.text2.tolist()
        self.target = frame.target.to_numpy(dtype=np.float32)
        self.category = frame.category1.tolist()

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[str, str, float, str]:
        return self.left[index], self.right[index], self.target[index], self.category[index]


def collate(tokenizer, max_length):
    def _collate(rows):
        left, right, targets, categories = zip(*rows)
        batch = tokenizer(list(left), list(right), padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        batch["targets"] = torch.tensor(targets, dtype=torch.float32)
        batch["categories"] = list(categories)
        return batch
    return _collate


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits[:, 0])
    return torch.softmax(logits, dim=-1)[:, 1]


def macro_pr_auc(model, loader, device) -> float:
    model.eval()
    scores, targets, categories = [], [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("targets")
            cats = batch.pop("categories")
            batch = {key: value.to(device) for key, value in batch.items()}
            scores.extend(probabilities(model(**batch).logits).float().cpu().tolist())
            targets.extend(labels.tolist())
            categories.extend(cats)
    values = []
    result = pd.DataFrame({"score": scores, "target": targets, "category": categories})
    for _, group in result.groupby("category"):
        # AP is undefined for one-class slices; omit only such tiny validation groups.
        if group.target.nunique() == 2:
            values.append(average_precision_score(group.target, group.score))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    frame = make_pairs(args.items_path, args.matches_path, args.max_train_rows, args.seed)
    train_frame, valid_frame = split_per_category(frame, args.validation_fraction, args.seed)
    print(f"Training pairs: {len(train_frame):,}; validation pairs: {len(valid_frame):,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True, trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code,
    ).to(device)
    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=device.type == "cuda",
                       collate_fn=collate(tokenizer, args.max_length))
    train_loader = DataLoader(PairDataset(train_frame), shuffle=True, **loader_args)
    valid_loader = DataLoader(PairDataset(valid_frame), shuffle=False, **loader_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 20), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -float("inf")
    output = Path(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            targets = batch.pop("targets").to(device)
            batch.pop("categories")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy(probabilities(model(**batch).logits), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        score = macro_pr_auc(model, valid_loader, device)
        print(f"Epoch {epoch}: validation macro PR-AUC = {score:.6f}")
        if score > best_score:
            best_score = score
            output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            (output / "training_metrics.txt").write_text(f"best_macro_pr_auc={score:.8f}\n", encoding="utf-8")
    print(f"Saved best model (macro PR-AUC {best_score:.6f}) to {output}")


if __name__ == "__main__":
    main()

