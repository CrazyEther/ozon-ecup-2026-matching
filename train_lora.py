"""LoRA smoke training for the GTE cross-encoder.

This is deliberately separate from ``train.py``: it saves a PEFT adapter, not
a replacement full model, and does not change the regular fine-tuning path.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from train import PairDataset, collate, macro_pr_auc, make_pairs, soft_binary_loss, split_per_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--matches-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def classifier_modules(model: torch.nn.Module) -> list[str]:
    """Keep a classification head trainable when PEFT excludes output layers."""
    suffixes = ("classifier", "score", "classification_head")
    return [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Module) and name.split(".")[-1] in suffixes
    ]


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
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True, trust_remote_code=args.trust_remote_code,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code,
    ).to(device)
    heads = classifier_modules(base_model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        modules_to_save=heads or None,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate(tokenizer, args.max_length),
    )
    train_loader = DataLoader(PairDataset(train_frame), shuffle=True, **loader_args)
    valid_loader = DataLoader(PairDataset(valid_frame), shuffle=False, **loader_args)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
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
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = soft_binary_loss(model(**batch).logits, targets)
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
            (output / "experiment.json").write_text(
                json.dumps({"base_model_path": args.model_path, "type": "lora_adapter"}, indent=2),
                encoding="utf-8",
            )
    print(f"Saved best LoRA adapter (macro PR-AUC {best_score:.6f}) to {output}")


if __name__ == "__main__":
    main()

