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
    parser.add_argument("--fatboost-model", default="",
                        help="Optional FatBoost .cbm; enables LoRA + FatBoost ensemble")
    parser.add_argument("--fatboost-features", default="",
                        help="FatBoost .features.joblib; inferred from model path when omitted")
    parser.add_argument("--fatboost-model-type", choices=["auto", "classifier", "ranker"],
                        default="auto")
    parser.add_argument("--ensemble-method", choices=["rank", "logit", "linear"], default="rank")
    parser.add_argument("--fatboost-weight", type=float, default=0.50,
                        help="Provisional default; select it with tune_ensemble.py")
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
    if args.fatboost_model:
        if not Path(args.fatboost_model).is_file():
            raise FileNotFoundError(f"FatBoost model {args.fatboost_model!r} is missing.")
        feature_path = args.fatboost_features or str(
            Path(args.fatboost_model).with_suffix(".features.joblib")
        )
        if not Path(feature_path).is_file():
            raise FileNotFoundError(f"FatBoost feature builder {feature_path!r} is missing.")
        from src.ensemble import predict_ensemble_to_csv
        predict_ensemble_to_csv(
            items_path=args.items_path,
            matches_path=args.matches_path,
            output_path=args.output_path,
            lora_model_path=args.model_path,
            lora_adapter_path=args.adapter_path or None,
            fatboost_model_path=args.fatboost_model,
            fatboost_feature_path=feature_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
            fatboost_weight=args.fatboost_weight,
            method=args.ensemble_method,
            fatboost_model_type=args.fatboost_model_type,
        )
        return
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
