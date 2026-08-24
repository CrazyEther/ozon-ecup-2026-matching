"""Download an initial checkpoint during development; never run in evaluation."""
from __future__ import annotations

import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Alibaba-NLP/gte-multilingual-reranker-base")
    parser.add_argument("--output", default="models/product-matcher")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(args.output)
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()

