"""LoRA training for the GTE cross-encoder.

The resulting artifact is a small PEFT adapter; it must be used together with
the local base GTE checkpoint during inference.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from train import (
    PairDataset, collate, make_pairs, probabilities, save_periodic_checkpoint,
    soft_binary_loss, split_per_category,
)


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
    parser.add_argument("--checkpoint-steps", type=int, default=500,
                        help="Save a recoverable LoRA adapter every N optimizer steps; 0 disables it")
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--resume-from", default="",
                        help="Path to a LoRA adapter checkpoint; optimizer starts fresh")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Disable automatic multi-GPU DDP even when several GPUs are available")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def classifier_modules(model: torch.nn.Module) -> list[str]:
    """Keep a classification head trainable when PEFT excludes output layers."""
    suffixes = ("classifier", "score", "classification_head")
    return [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Module) and name.split(".")[-1] in suffixes
    ]


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without padding or duplicated examples."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = range(rank, size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def maybe_launch_distributed(args: argparse.Namespace) -> bool:
    """Relaunch under torchrun so the normal CLI automatically uses every GPU."""
    already_launched = int(os.environ.get("WORLD_SIZE", "1")) > 1
    gpu_count = torch.cuda.device_count()
    if already_launched or args.single_gpu or gpu_count < 2:
        return False
    print(f"Detected {gpu_count} GPUs; launching one DDP worker per GPU.", flush=True)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    subprocess.run(command, check=True)
    return True


def distributed_macro_pr_auc(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
) -> float:
    """Evaluate all validation rows across ranks and return the exact global score."""
    model.eval()
    scores: list[float] = []
    targets: list[float] = []
    categories: list[object] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("targets")
            cats = batch.pop("categories")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            scores.extend(probabilities(model(**batch).logits).float().cpu().tolist())
            targets.extend(labels.tolist())
            categories.extend(cats)

    payload = (scores, targets, categories)
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
        if rank == 0:
            scores = [value for part in gathered for value in part[0]]
            targets = [value for part in gathered for value in part[1]]
            categories = [value for part in gathered for value in part[2]]

    score = 0.0
    if rank == 0:
        grouped: dict[object, tuple[list[float], list[float]]] = {}
        for prediction, target, category in zip(scores, targets, categories):
            group_scores, group_targets = grouped.setdefault(category, ([], []))
            group_scores.append(prediction)
            group_targets.append(target)
        values = [
            average_precision_score(group_targets, group_scores)
            for group_scores, group_targets in grouped.values()
            if len(set(group_targets)) == 2
        ]
        score = float(np.mean(values)) if values else float("nan")
    if world_size > 1:
        score_tensor = torch.tensor(score, dtype=torch.float64, device=device)
        dist.broadcast(score_tensor, src=0)
        score = float(score_tensor.item())
    return score


def main() -> None:
    args = parse_args()
    if maybe_launch_distributed(args):
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU DDP was requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    is_main = rank == 0

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    frame = make_pairs(args.items_path, args.matches_path, args.max_train_rows, args.seed)
    train_frame, valid_frame = split_per_category(frame, args.validation_fraction, args.seed)
    if is_main:
        print(f"Training pairs: {len(train_frame):,}; validation pairs: {len(valid_frame):,}")
        if distributed:
            print(
                f"DDP enabled on {world_size} GPUs; effective global batch size: "
                f"{args.batch_size * world_size:,}",
            )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True, trust_remote_code=args.trust_remote_code,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code,
    ).to(device)
    if args.resume_from:
        print(f"Resuming LoRA adapter from {args.resume_from}; optimizer will start fresh.")
        model = PeftModel.from_pretrained(base_model, args.resume_from, is_trainable=True, local_files_only=True)
    else:
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
    if is_main:
        model.print_trainable_parameters()

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate(tokenizer, args.max_length),
    )
    train_dataset = PairDataset(train_frame)
    valid_dataset = PairDataset(valid_frame)
    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
        )
        if distributed else None
    )
    valid_sampler = (
        DistributedEvalSampler(len(valid_dataset), rank, world_size)
        if distributed else None
    )
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=train_sampler is None, **loader_args,
    )
    valid_loader = DataLoader(
        valid_dataset, sampler=valid_sampler, shuffle=False, **loader_args,
    )
    trainable_model = model
    if distributed:
        model = DistributedDataParallel(
            trainable_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
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
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)
        for step_in_epoch, batch in enumerate(progress, start=1):
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
            global_step += 1
            if is_main:
                progress.set_postfix(loss=f"{loss.item():.4f}")
            if args.checkpoint_steps and global_step % args.checkpoint_steps == 0:
                if is_main:
                    save_periodic_checkpoint(
                        trainable_model, tokenizer, output, global_step, epoch, step_in_epoch,
                        args.keep_checkpoints,
                    )
                if distributed:
                    dist.barrier()
        score = distributed_macro_pr_auc(
            trainable_model, valid_loader, device, rank, world_size,
        )
        if is_main:
            print(f"Epoch {epoch}: validation macro PR-AUC = {score:.6f}")
        if is_main and score > best_score:
            best_score = score
            trainable_model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            (output / "experiment.json").write_text(
                json.dumps({"base_model_path": args.model_path, "type": "lora_adapter"}, indent=2),
                encoding="utf-8",
            )
        if distributed:
            dist.barrier()
    if is_main:
        print(f"Saved best LoRA adapter (macro PR-AUC {best_score:.6f}) to {output}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
