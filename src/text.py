"""Deterministic product-card serialization shared by train and inference."""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

_SPACE = re.compile(r"\s+")


def _as_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return _SPACE.sub(" ", str(value)).strip()


def attributes_to_text(value: Any) -> str:
    """Turn folded JSON attributes into stable, readable text.

    Invalid attributes are intentionally retained as plain text: silently
    deleting them would turn malformed but informative production rows empty.
    """
    raw = _as_text(value)
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if not isinstance(parsed, dict):
        return raw
    parts: list[str] = []
    for key, item in parsed.items():
        key_text, item_text = _as_text(key), _as_text(item)
        if key_text and item_text:
            parts.append(f"{key_text}: {item_text}")
    return "; ".join(parts)


def product_text(name: Any, attributes: Any, category: Any) -> str:
    fields = [f"Название: {_as_text(name)}", f"Категория: {_as_text(category)}"]
    attrs = attributes_to_text(attributes)
    if attrs:
        fields.append(f"Характеристики: {attrs}")
    return " [SEP] ".join(fields)


def add_product_text(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "name", "attributes", "category"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Items parquet is missing columns: {sorted(missing)}")
    result = frame.loc[:, ["id", "name", "attributes", "category"]].copy()
    result["text"] = [
        product_text(name, attrs, category)
        for name, attrs, category in zip(result.name, result.attributes, result.category)
    ]
    return result.loc[:, ["id", "text", "category"]]

