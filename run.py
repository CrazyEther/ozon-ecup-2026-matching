"""Offline competition entry point for Ozon product matching."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.inference import predict_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--items_path", "--items-path", required=True)
    parser.add_argument("-m", "--matches_path", "--matches-path", required=True)
    parser.add_argument("-o", "--output_path", "--output-path", required=True)
    parser.add_argument("--model-path", default="models/product-matcher")
    parser.add_argument("--adapter-path", default="",
                        help="Optional local LoRA adapter directory trained by train_lora.py")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Required by the bundled GTE reranker implementation")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(
            f"Model directory {args.model_path!r} is missing. "
            "Bundle a local checkpoint in models/product-matcher."
        )
    if args.adapter_path and not Path(args.adapter_path).is_dir():
        raise FileNotFoundError(f"LoRA adapter directory {args.adapter_path!r} is missing.")
    predict_to_csv(
        items_path=args.items_path,
        matches_path=args.matches_path,
        output_path=args.output_path,
        model_path=args.model_path,
        adapter_path=args.adapter_path or None,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()

