"""High-signal lexical features for product matching.

They are intentionally lightweight: the feature model is a fast companion to a
cross-encoder, and also provides a reliable CPU-only fallback in the checker.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

_NON_WORD = re.compile(r"[^\w]+", flags=re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)(?:\s*(?:x|х|×)\s*(\d+(?:[.,]\d+)?))?", flags=re.I)


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value)


def normalize(value: Any) -> str:
    return " ".join(_NON_WORD.sub(" ", _text(value).casefold()).split())


def _numbers(text: str) -> frozenset[str]:
    found: set[str] = set()
    for first, second in _NUMBER.findall(text):
        found.add(first.replace(",", "."))
        if second:
            found.add(second.replace(",", "."))
    return frozenset(found)


def _attributes(value: Any) -> dict[str, str]:
    try:
        raw = json.loads(_text(value))
    except (TypeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {normalize(key): normalize(item) for key, item in raw.items() if normalize(key) and normalize(item)}


@dataclass(frozen=True)
class Card:
    name: str
    all_text: str
    name_tokens: frozenset[str]
    all_tokens: frozenset[str]
    numbers: frozenset[str]
    attrs: dict[str, str]


def make_card(name: Any, attributes: Any) -> Card:
    name_text = normalize(name)
    attrs = _attributes(attributes)
    attrs_text = " ".join(f"{key} {value}" for key, value in attrs.items())
    all_text = f"{name_text} {attrs_text}".strip()
    return Card(name_text, all_text, frozenset(name_text.split()), frozenset(all_text.split()),
                _numbers(all_text), attrs)


def _jaccard(first: frozenset[str] | set[str], second: frozenset[str] | set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def features_for_cards(first: Card, second: Card) -> list[float]:
    common_keys = set(first.attrs) & set(second.attrs)
    value_matches = sum(first.attrs[key] == second.attrs[key] for key in common_keys)
    attr_values_1 = frozenset(first.attrs.values())
    attr_values_2 = frozenset(second.attrs.values())
    has_numbers = bool(first.numbers) and bool(second.numbers)
    numeric_overlap = _jaccard(first.numbers, second.numbers) if has_numbers else 0.0
    return [
        float(first.name == second.name),
        fuzz.ratio(first.name, second.name) / 100.0,
        fuzz.token_set_ratio(first.name, second.name) / 100.0,
        fuzz.ratio(first.all_text, second.all_text) / 100.0,
        fuzz.token_set_ratio(first.all_text, second.all_text) / 100.0,
        _jaccard(first.name_tokens, second.name_tokens),
        _jaccard(first.all_tokens, second.all_tokens),
        _jaccard(set(first.attrs), set(second.attrs)),
        value_matches / len(common_keys) if common_keys else 0.0,
        _jaccard(attr_values_1, attr_values_2),
        float(has_numbers),
        numeric_overlap,
        float(has_numbers and not (first.numbers & second.numbers)),
        min(len(first.name), len(second.name)) / max(1, max(len(first.name), len(second.name))),
    ]


FEATURE_NAMES = [
    "name_exact", "name_ratio", "name_token_set", "text_ratio", "text_token_set",
    "name_jaccard", "text_jaccard", "attribute_key_jaccard", "shared_attribute_value_rate",
    "attribute_value_jaccard", "both_have_numbers", "number_jaccard", "number_conflict",
    "name_length_ratio",
]


def pair_features(items: pd.DataFrame, matches: pd.DataFrame) -> np.ndarray:
    required_items = {"id", "name", "attributes"}
    required_matches = {"id1", "id2"}
    if missing := required_items.difference(items.columns):
        raise ValueError(f"items is missing {sorted(missing)}")
    if missing := required_matches.difference(matches.columns):
        raise ValueError(f"matches is missing {sorted(missing)}")
    cards = {row.id: make_card(row.name, row.attributes) for row in items.itertuples(index=False)}
    output = np.zeros((len(matches), len(FEATURE_NAMES)), dtype=np.float32)
    for row_index, pair in enumerate(matches.loc[:, ["id1", "id2"]].itertuples(index=False)):
        first, second = cards.get(pair.id1), cards.get(pair.id2)
        if first is not None and second is not None:
            output[row_index] = features_for_cards(first, second)
    return output

