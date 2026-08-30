"""Merge a trained LoRA adapter into GTE and create an offline checkpoint."""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def copy_remote_code(model, output: Path) -> None:
    """Bundle cached trust_remote_code modules and make auto_map local."""
    source_dirs = {
        Path(inspect.getfile(type(model))).resolve().parent,
        Path(inspect.getfile(type(model.config))).resolve().parent,
    }
    copied = []
    for source_dir in source_dirs:
        for source in source_dir.rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            destination = output / source.relative_to(source_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination.relative_to(output).as_posix())

    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    auto_map = config.get("auto_map", {})
    config["auto_map"] = {
        key: value.split("--", 1)[-1] if isinstance(value, str) else value
        for key, value in auto_map.items()
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "bundled_code.json").write_text(
        json.dumps({"python_files": sorted(set(copied))}, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=args.trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(
        base_model, args.adapter, local_files_only=True,
    )
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    merged_model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    copy_remote_code(merged_model, output)
    (output / "finalization.json").write_text(
        json.dumps(
            {"base_model": args.base_model, "adapter": args.adapter, "merged": True},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Saved merged offline model to {output}")


if __name__ == "__main__":
    main()

