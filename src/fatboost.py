"""Feature pipeline for the CatBoost product matcher.

The builder deliberately mixes complementary signals:

* deterministic product rules (model codes, quantities, brands, attributes);
* word and character TF-IDF cosine similarities;
* native CatBoost categorical and text columns;
* optional tiny-transformer embedding distances.

The fitted builder is serializable with joblib and must be shipped next to the
CatBoost model.  No target is used while fitting the TF-IDF vectorizers.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


EMPTY_TEXT = "__empty__"
CAT_FEATURES = ["category_1", "category_2", "category_pair", "brand_pair"]
TEXT_FEATURES = ["pair_name_text", "shared_text", "difference_text"]

_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w.+%/×хx-]+", flags=re.UNICODE)
_MODEL_TOKEN = re.compile(r"(?<!\w)[\w]+(?:[-./][\w]+)*(?!\w)", flags=re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)(?!\w)")
_PACK = re.compile(
    r"(?:\b(?:упак(?:овка)?|комплект|набор)\s*(?:из\s*)?(\d+)|"
    r"\b(\d+)\s*(?:шт|штук|pcs)\b|(?:^|\s)[xх×]\s*(\d+)\b)",
    flags=re.I,
)
_QUANTITY = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)\s*"
    r"(мг|mg|кг|kg|гр|г|g|мл|ml|cl|л|l|мм|mm|см|cm|м|m|"
    r"вт|w|квт|kw|мвт|mw|кб|kb|мб|mb|гб|gb|тб|tb|"
    r"гц|hz|кгц|khz|мгц|mhz|ггц|ghz|дюйм(?:а|ов)?|inch(?:es)?|\")(?=\s|$|[,;:)])",
    flags=re.I,
)

# dimension, multiplier to a stable base unit
_UNITS: dict[str, tuple[str, float]] = {
    "мг": ("mass_mg", 1.0), "mg": ("mass_mg", 1.0),
    "г": ("mass_mg", 1_000.0), "гр": ("mass_mg", 1_000.0), "g": ("mass_mg", 1_000.0),
    "кг": ("mass_mg", 1_000_000.0), "kg": ("mass_mg", 1_000_000.0),
    "мл": ("volume_ml", 1.0), "ml": ("volume_ml", 1.0), "cl": ("volume_ml", 10.0),
    "л": ("volume_ml", 1_000.0), "l": ("volume_ml", 1_000.0),
    "мм": ("length_mm", 1.0), "mm": ("length_mm", 1.0),
    "см": ("length_mm", 10.0), "cm": ("length_mm", 10.0),
    "м": ("length_mm", 1_000.0), "m": ("length_mm", 1_000.0),
    "вт": ("power_w", 1.0), "w": ("power_w", 1.0),
    "квт": ("power_w", 1_000.0), "kw": ("power_w", 1_000.0),
    "мвт": ("power_w", 1_000_000.0), "mw": ("power_w", 1_000_000.0),
    "кб": ("storage_mb", 1.0 / 1_024.0), "kb": ("storage_mb", 1.0 / 1_024.0),
    "мб": ("storage_mb", 1.0), "mb": ("storage_mb", 1.0),
    "гб": ("storage_mb", 1_024.0), "gb": ("storage_mb", 1_024.0),
    "тб": ("storage_mb", 1_048_576.0), "tb": ("storage_mb", 1_048_576.0),
    "гц": ("frequency_hz", 1.0), "hz": ("frequency_hz", 1.0),
    "кгц": ("frequency_hz", 1_000.0), "khz": ("frequency_hz", 1_000.0),
    "мгц": ("frequency_hz", 1_000_000.0), "mhz": ("frequency_hz", 1_000_000.0),
    "ггц": ("frequency_hz", 1_000_000_000.0), "ghz": ("frequency_hz", 1_000_000_000.0),
    "дюйм": ("diagonal_inch", 1.0), "дюйма": ("diagonal_inch", 1.0),
    "дюймов": ("diagonal_inch", 1.0), "inch": ("diagonal_inch", 1.0),
    "inches": ("diagonal_inch", 1.0), '"': ("diagonal_inch", 1.0),
}
_BRAND_KEYS = ("бренд", "brand", "марка", "производитель", "manufacturer")
_SKU_KEYS = (
    "артикул", "sku", "модель", "model", "код товара", "код производителя",
    "part number", "mpn", "ean", "gtin", "barcode", "штрихкод",
)
_CODE_CONFUSABLES = str.maketrans({
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "к": "k",
    "м": "m", "о": "o", "р": "p", "т": "t", "х": "x", "у": "y",
})


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if not isinstance(value, (list, dict)) and pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _as_text(value)).casefold().replace("ё", "е")
    return _SPACE.sub(" ", _NON_WORD.sub(" ", text)).strip()


def _flatten_json(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = normalize(key)
            next_prefix = f"{prefix} {key_text}".strip()
            yield from _flatten_json(item, next_prefix)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_json(item, prefix)
    else:
        item_text = normalize(value)
        if prefix and item_text:
            yield prefix, item_text


def parse_attributes(value: Any) -> dict[str, str]:
    raw = _as_text(value)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"attributes": normalize(raw)} if normalize(raw) else {}
    if not isinstance(parsed, dict):
        return {"attributes": normalize(raw)} if normalize(raw) else {}
    combined: dict[str, list[str]] = {}
    for key, item in _flatten_json(parsed):
        combined.setdefault(key, []).append(item)
    return {key: " ".join(values) for key, values in combined.items()}


def _model_code(token: str) -> str:
    return re.sub(r"[-./_]", "", token.casefold().replace("ё", "е").translate(_CODE_CONFUSABLES))


def extract_model_codes(text: str) -> frozenset[str]:
    result: set[str] = set()
    for token in _MODEL_TOKEN.findall(text):
        compact = _model_code(token)
        if 3 <= len(compact) <= 40 and any(char.isalpha() for char in compact) and any(char.isdigit() for char in compact):
            # Avoid treating a normalized unit such as 500ml as a discriminative SKU.
            if not any(compact.endswith(unit) for unit in ("mg", "kg", "ml", "mm", "cm", "gb", "tb", "mhz", "ghz")):
                result.add(compact)
    return frozenset(result)


def extract_quantities(text: str) -> dict[str, tuple[float, ...]]:
    values: dict[str, list[float]] = {}
    for raw_value, raw_unit in _QUANTITY.findall(text):
        unit = raw_unit.casefold()
        dimension_scale = _UNITS.get(unit)
        if dimension_scale is None:
            continue
        dimension, scale = dimension_scale
        try:
            value = float(raw_value.replace(",", ".")) * scale
        except ValueError:
            continue
        values.setdefault(dimension, []).append(value)
    return {key: tuple(sorted(items)) for key, items in values.items()}


def extract_pack_sizes(text: str) -> frozenset[int]:
    result: set[int] = set()
    for match in _PACK.findall(text):
        for value in match:
            if value:
                parsed = int(value)
                if 1 < parsed <= 10_000:
                    result.add(parsed)
    return frozenset(result)


def _values_for_keys(attrs: dict[str, str], fragments: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        value for key, value in attrs.items()
        if any(fragment in key for fragment in fragments) and value
    )


@dataclass(frozen=True)
class ProductCard:
    name: str
    all_text: str
    name_tokens: frozenset[str]
    all_tokens: frozenset[str]
    attrs: dict[str, str]
    numbers: frozenset[str]
    model_codes: frozenset[str]
    quantities: dict[str, tuple[float, ...]]
    pack_sizes: frozenset[int]
    brands: frozenset[str]
    sku_values: frozenset[str]
    category: str


def make_card(name: Any, attributes: Any, category: Any = "") -> ProductCard:
    name_text = normalize(name)
    attrs = parse_attributes(attributes)
    attrs_text = " ".join(f"{key} {value}" for key, value in attrs.items())
    all_text = f"{name_text} {attrs_text}".strip()
    numbers = frozenset(value.replace(",", ".") for value in _NUMBER.findall(all_text))
    sku_values = _values_for_keys(attrs, _SKU_KEYS)
    codes = set(extract_model_codes(all_text))
    for value in sku_values:
        compact = _model_code(value)
        if compact:
            codes.add(compact)
    return ProductCard(
        name=name_text,
        all_text=all_text,
        name_tokens=frozenset(name_text.split()),
        all_tokens=frozenset(all_text.split()),
        attrs=attrs,
        numbers=numbers,
        model_codes=frozenset(codes),
        quantities=extract_quantities(all_text),
        pack_sizes=extract_pack_sizes(all_text),
        brands=_values_for_keys(attrs, _BRAND_KEYS),
        sku_values=sku_values,
        category=normalize(category) or "__missing__",
    )


def _jaccard(first: frozenset[Any] | set[Any], second: frozenset[Any] | set[Any]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _containment(first: frozenset[Any], second: frozenset[Any]) -> float:
    return len(first & second) / len(first) if first else float(not second)


def _best_fuzzy(first: frozenset[str], second: frozenset[str]) -> float:
    if not first or not second:
        return 0.0
    return max(fuzz.ratio(left, right) for left in first for right in second) / 100.0


def _quantity_features(first: ProductCard, second: ProductCard) -> tuple[float, float, float, float]:
    shared_dimensions = set(first.quantities) & set(second.quantities)
    if not shared_dimensions:
        return 0.0, 0.0, 0.0, 0.0
    relative_distances: list[float] = []
    exact = 0
    for dimension in shared_dimensions:
        best = min(
            abs(left - right) / max(abs(left), abs(right), 1e-9)
            for left in first.quantities[dimension]
            for right in second.quantities[dimension]
        )
        relative_distances.append(min(best, 1.0))
        exact += best <= 0.01
    mean_distance = float(np.mean(relative_distances))
    return (
        len(shared_dimensions) / max(1, len(set(first.quantities) | set(second.quantities))),
        exact / len(shared_dimensions),
        mean_distance,
        float(any(value > 0.05 for value in relative_distances)),
    )


NUMERIC_FEATURES = [
    "name_exact", "name_ratio", "name_wratio", "name_partial_ratio", "name_token_sort",
    "name_token_set", "text_ratio", "text_wratio", "text_partial_ratio", "text_token_sort",
    "text_token_set", "name_jaccard", "name_containment_1", "name_containment_2",
    "text_jaccard", "text_containment_1", "text_containment_2", "attribute_key_jaccard",
    "attribute_key_containment_1", "attribute_key_containment_2", "shared_attribute_count",
    "shared_attribute_value_rate", "attribute_conflict_rate", "attribute_value_jaccard",
    "both_have_numbers", "number_jaccard", "number_conflict", "model_code_jaccard",
    "model_code_overlap", "model_code_conflict", "model_code_best_ratio", "sku_exact_overlap",
    "brand_jaccard", "brand_conflict", "quantity_dimension_jaccard", "quantity_exact_rate",
    "quantity_relative_distance", "quantity_conflict", "pack_jaccard", "pack_conflict",
    "name_length_ratio", "name_token_count_ratio", "text_length_ratio", "same_category",
    "word_tfidf_cosine", "char_tfidf_cosine",
]
EMBEDDING_FEATURES = [
    "embedding_cosine", "embedding_l1_mean", "embedding_l2_rms",
    "embedding_max_abs", "embedding_hadamard_mean",
]


def numeric_features(first: ProductCard, second: ProductCard) -> list[float]:
    common_keys = set(first.attrs) & set(second.attrs)
    value_matches = sum(first.attrs[key] == second.attrs[key] for key in common_keys)
    conflicts = len(common_keys) - value_matches
    attr_values_1, attr_values_2 = frozenset(first.attrs.values()), frozenset(second.attrs.values())
    has_numbers = bool(first.numbers) and bool(second.numbers)
    model_overlap = first.model_codes & second.model_codes
    quantity = _quantity_features(first, second)
    both_pack = bool(first.pack_sizes) and bool(second.pack_sizes)
    return [
        float(first.name == second.name),
        fuzz.ratio(first.name, second.name) / 100.0,
        fuzz.WRatio(first.name, second.name) / 100.0,
        fuzz.partial_ratio(first.name, second.name) / 100.0,
        fuzz.token_sort_ratio(first.name, second.name) / 100.0,
        fuzz.token_set_ratio(first.name, second.name) / 100.0,
        fuzz.ratio(first.all_text, second.all_text) / 100.0,
        fuzz.WRatio(first.all_text, second.all_text) / 100.0,
        fuzz.partial_ratio(first.all_text, second.all_text) / 100.0,
        fuzz.token_sort_ratio(first.all_text, second.all_text) / 100.0,
        fuzz.token_set_ratio(first.all_text, second.all_text) / 100.0,
        _jaccard(first.name_tokens, second.name_tokens),
        _containment(first.name_tokens, second.name_tokens),
        _containment(second.name_tokens, first.name_tokens),
        _jaccard(first.all_tokens, second.all_tokens),
        _containment(first.all_tokens, second.all_tokens),
        _containment(second.all_tokens, first.all_tokens),
        _jaccard(set(first.attrs), set(second.attrs)),
        len(common_keys) / max(1, len(first.attrs)),
        len(common_keys) / max(1, len(second.attrs)),
        float(len(common_keys)),
        value_matches / len(common_keys) if common_keys else 0.0,
        conflicts / len(common_keys) if common_keys else 0.0,
        _jaccard(attr_values_1, attr_values_2),
        float(has_numbers),
        _jaccard(first.numbers, second.numbers) if has_numbers else 0.0,
        float(has_numbers and not (first.numbers & second.numbers)),
        _jaccard(first.model_codes, second.model_codes) if first.model_codes or second.model_codes else 0.0,
        float(bool(model_overlap)),
        float(bool(first.model_codes) and bool(second.model_codes) and first.model_codes != second.model_codes),
        _best_fuzzy(first.model_codes, second.model_codes),
        float(bool(first.sku_values & second.sku_values)),
        _jaccard(first.brands, second.brands) if first.brands or second.brands else 0.0,
        float(bool(first.brands) and bool(second.brands) and not (first.brands & second.brands)),
        *quantity,
        _jaccard(first.pack_sizes, second.pack_sizes) if both_pack else 0.0,
        float(both_pack and not (first.pack_sizes & second.pack_sizes)),
        min(len(first.name), len(second.name)) / max(1, max(len(first.name), len(second.name))),
        min(len(first.name_tokens), len(second.name_tokens)) / max(1, max(len(first.name_tokens), len(second.name_tokens))),
        min(len(first.all_text), len(second.all_text)) / max(1, max(len(first.all_text), len(second.all_text))),
        float(first.category == second.category),
    ]


def _pair_text_fields(first: ProductCard, second: ProductCard) -> tuple[str, str, str]:
    shared = sorted(first.all_tokens & second.all_tokens)
    left_only = sorted(first.all_tokens - second.all_tokens)
    right_only = sorted(second.all_tokens - first.all_tokens)
    conflicts = [
        f"{key}_{first.attrs[key]}_vs_{second.attrs[key]}"
        for key in sorted(set(first.attrs) & set(second.attrs))
        if first.attrs[key] != second.attrs[key]
    ]
    pair_name = f"left {first.name} pair right {second.name}".strip() or EMPTY_TEXT
    # Bounds keep native text processing predictable on cards with enormous JSON.
    shared_text = " ".join(shared[:96]) or EMPTY_TEXT
    difference = " ".join([*(f"left_{token}" for token in left_only[:64]),
                           *(f"right_{token}" for token in right_only[:64]), *conflicts[:32]])
    return pair_name, shared_text, difference or EMPTY_TEXT


class TinyTextEncoder:
    """Mean-pooling encoder using only Transformers; sentence-transformers is unnecessary."""

    def __init__(
        self, model_name_or_path: str, batch_size: int = 256, max_length: int = 96,
        device: str | None = None, local_files_only: bool = True, pooling: str = "cls",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, local_files_only=local_files_only, use_fast=True,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only, trust_remote_code=False,
        ).to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        torch = self.torch
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start:start + self.batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            chunks.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(chunks) if chunks else np.empty((0, 0), dtype=np.float32)

    def save_pretrained(self, output: str | Path) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output)
        self.model.save_pretrained(output, safe_serialization=True)


@dataclass
class FatBoostFeatureBuilder:
    word_max_features: int = 120_000
    char_max_features: int = 160_000
    min_df: int = 2
    native_text: bool = True
    embedding_model: str | None = None
    embedding_pooling: str = "cls"

    def __post_init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            lowercase=False, ngram_range=(1, 2), min_df=self.min_df,
            max_features=self.word_max_features, sublinear_tf=True, dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", lowercase=False, ngram_range=(3, 5), min_df=self.min_df,
            max_features=self.char_max_features, sublinear_tf=True, dtype=np.float32,
        )
        self.fitted = False

    @property
    def numeric_feature_names(self) -> list[str]:
        return [*NUMERIC_FEATURES, *(EMBEDDING_FEATURES if self.embedding_model else [])]

    @property
    def cat_feature_names(self) -> list[str]:
        return CAT_FEATURES

    @property
    def text_feature_names(self) -> list[str]:
        return TEXT_FEATURES if self.native_text else []

    def _cards(self, items: pd.DataFrame) -> tuple[list[Any], list[ProductCard]]:
        required = {"id", "name", "attributes", "category"}
        if missing := required.difference(items.columns):
            raise ValueError(f"items is missing {sorted(missing)}")
        ids: list[Any] = []
        cards: list[ProductCard] = []
        for row in items.loc[:, ["id", "name", "attributes", "category"]].itertuples(index=False):
            ids.append(row.id)
            cards.append(make_card(row.name, row.attributes, row.category))
        return ids, cards

    def fit(self, items: pd.DataFrame) -> "FatBoostFeatureBuilder":
        _, cards = self._cards(items)
        texts = [card.all_text or EMPTY_TEXT for card in cards]
        names = [card.name or EMPTY_TEXT for card in cards]
        self.word_vectorizer.fit(texts)
        self.char_vectorizer.fit(names)
        self.fitted = True
        return self

    @staticmethod
    def _cosine(matrix: Any, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        output = np.zeros(len(left), dtype=np.float32)
        valid = (left >= 0) & (right >= 0)
        valid_indices = np.flatnonzero(valid)
        # Keep temporary sparse matrices bounded for large pair files.
        for start in range(0, len(valid_indices), 100_000):
            pair_indices = valid_indices[start:start + 100_000]
            products = matrix[left[pair_indices]].multiply(matrix[right[pair_indices]])
            output[pair_indices] = np.asarray(products.sum(axis=1)).ravel()
        return output

    @staticmethod
    def _embedding_pair_features(
        embeddings: np.ndarray, left: np.ndarray, right: np.ndarray,
    ) -> np.ndarray:
        output = np.zeros((len(left), len(EMBEDDING_FEATURES)), dtype=np.float32)
        valid = (left >= 0) & (right >= 0)
        valid_indices = np.flatnonzero(valid)
        for start in range(0, len(valid_indices), 50_000):
            pair_indices = valid_indices[start:start + 50_000]
            one, two = embeddings[left[pair_indices]], embeddings[right[pair_indices]]
            delta = np.abs(one - two)
            output[pair_indices, 0] = np.sum(one * two, axis=1)
            output[pair_indices, 1] = np.mean(delta, axis=1)
            output[pair_indices, 2] = np.sqrt(np.mean(np.square(delta), axis=1))
            output[pair_indices, 3] = np.max(delta, axis=1)
            output[pair_indices, 4] = np.mean(one * two, axis=1)
        return output

    def transform(
        self, items: pd.DataFrame, matches: pd.DataFrame,
        encoder: TinyTextEncoder | None = None,
    ) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FatBoostFeatureBuilder.fit must be called before transform")
        if missing := {"id1", "id2"}.difference(matches.columns):
            raise ValueError(f"matches is missing {sorted(missing)}")
        ids, cards = self._cards(items)
        id_to_index = {item_id: index for index, item_id in enumerate(ids)}
        left = matches.id1.map(id_to_index).fillna(-1).to_numpy(dtype=np.int64)
        right = matches.id2.map(id_to_index).fillna(-1).to_numpy(dtype=np.int64)

        base = np.zeros((len(matches), len(NUMERIC_FEATURES) - 2), dtype=np.float32)
        pair_text: list[tuple[str, str, str]] = []
        missing_card = make_card("", "", "__missing__")
        for row_index, (left_index, right_index) in enumerate(zip(left, right)):
            first = cards[left_index] if left_index >= 0 else missing_card
            second = cards[right_index] if right_index >= 0 else missing_card
            base[row_index] = numeric_features(first, second)
            if self.native_text:
                pair_text.append(_pair_text_fields(first, second))

        texts = [card.all_text or EMPTY_TEXT for card in cards]
        names = [card.name or EMPTY_TEXT for card in cards]
        word_matrix = self.word_vectorizer.transform(texts)
        char_matrix = self.char_vectorizer.transform(names)
        tfidf = np.column_stack([
            self._cosine(word_matrix, left, right), self._cosine(char_matrix, left, right),
        ]).astype(np.float32, copy=False)
        numeric = np.column_stack([base, tfidf])

        if self.embedding_model:
            if encoder is None:
                raise ValueError("This feature builder requires a TinyTextEncoder")
            embeddings = encoder.encode(texts)
            numeric = np.column_stack([
                numeric, self._embedding_pair_features(embeddings, left, right),
            ])

        result = pd.DataFrame(numeric, columns=self.numeric_feature_names)
        left_cards = [cards[index] if index >= 0 else missing_card for index in left]
        right_cards = [cards[index] if index >= 0 else missing_card for index in right]
        result["category_1"] = [card.category for card in left_cards]
        result["category_2"] = [card.category for card in right_cards]
        result["category_pair"] = [f"{one.category}|||{two.category}" for one, two in zip(left_cards, right_cards)]
        result["brand_pair"] = [
            f"{'|'.join(sorted(one.brands)) or '__missing__'}|||{'|'.join(sorted(two.brands)) or '__missing__'}"
            for one, two in zip(left_cards, right_cards)
        ]
        if self.native_text:
            for column, position in zip(TEXT_FEATURES, range(3)):
                result[column] = [fields[position] for fields in pair_text]
        return result

    def fit_transform(
        self, items: pd.DataFrame, matches: pd.DataFrame,
        encoder: TinyTextEncoder | None = None,
    ) -> pd.DataFrame:
        return self.fit(items).transform(items, matches, encoder=encoder)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output.astype(np.float32)
