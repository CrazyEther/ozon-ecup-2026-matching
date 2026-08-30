from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.fatboost import (
    NUMERIC_FEATURES,
    FatBoostFeatureBuilder,
    extract_model_codes,
    extract_pack_sizes,
    extract_quantities,
    make_card,
    numeric_features,
)


class FatBoostFeaturesTest(unittest.TestCase):
    def test_model_codes_handle_cyrillic_confusables(self) -> None:
        self.assertEqual(extract_model_codes("Samsung SM-S918B"), frozenset({"sms918b"}))
        self.assertEqual(extract_model_codes("Samsung SМ-S918В"), frozenset({"sms918b"}))

    def test_quantities_are_normalized(self) -> None:
        self.assertEqual(extract_quantities("Вода 0,5 л")["volume_ml"], (500.0,))
        self.assertEqual(extract_quantities("Вода 500 мл")["volume_ml"], (500.0,))
        self.assertEqual(extract_quantities("SSD 1 ТБ")["storage_mb"], (1_048_576.0,))

    def test_pack_sizes(self) -> None:
        self.assertEqual(extract_pack_sizes("упаковка из 12 шт"), frozenset({12}))

    def test_numeric_feature_shape_and_signal(self) -> None:
        one = make_card(
            "Samsung S23 SM-S911B 256 GB",
            '{"Бренд":"Samsung","Модель":"SM-S911B","Вес":"500 г"}',
            "phones",
        )
        same = make_card(
            "Смартфон Samsung S23 SM-S911B 256GB",
            '{"brand":"Samsung","model":"SM-S911B","Вес":"0.5 кг"}',
            "phones",
        )
        other = make_card(
            "Samsung S23 Plus SM-S916B 256GB",
            '{"brand":"Samsung","model":"SM-S916B","Вес":"0.5 кг"}',
            "phones",
        )
        same_features = numeric_features(one, same)
        other_features = numeric_features(one, other)
        self.assertEqual(len(same_features), len(NUMERIC_FEATURES) - 2)
        index = {name: position for position, name in enumerate(NUMERIC_FEATURES[:-2])}
        self.assertEqual(same_features[index["model_code_overlap"]], 1.0)
        self.assertEqual(other_features[index["model_code_conflict"]], 1.0)
        self.assertEqual(same_features[index["quantity_exact_rate"]], 1.0)

    def test_builder_produces_finite_tfidf_features(self) -> None:
        items = pd.DataFrame([
            {"id": 1, "name": "Молоко 1 л Домик", "attributes": '{"бренд":"Домик"}', "category": "food"},
            {"id": 2, "name": "Молоко Домик 1000 мл", "attributes": '{"brand":"Домик"}', "category": "food"},
            {"id": 3, "name": "Телефон A55", "attributes": '{"model":"A55"}', "category": "phones"},
        ])
        pairs = pd.DataFrame([{"id1": 1, "id2": 2}, {"id1": 1, "id2": 3}])
        builder = FatBoostFeatureBuilder(min_df=1, word_max_features=100, char_max_features=100)
        result = builder.fit_transform(items, pairs)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isfinite(result[builder.numeric_feature_names].to_numpy()).all())
        self.assertGreater(result.loc[0, "word_tfidf_cosine"], result.loc[1, "word_tfidf_cosine"])
        self.assertGreater(result.loc[0, "char_tfidf_cosine"], result.loc[1, "char_tfidf_cosine"])

    def test_optional_embedding_branch(self) -> None:
        class FakeEncoder:
            def encode(self, texts: list[str]) -> np.ndarray:
                values = np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4) + 1
                return values / np.linalg.norm(values, axis=1, keepdims=True)

        items = pd.DataFrame([
            {"id": 1, "name": "товар один", "attributes": "{}", "category": "cat"},
            {"id": 2, "name": "товар два", "attributes": "{}", "category": "cat"},
        ])
        pairs = pd.DataFrame([{"id1": 1, "id2": 2}])
        builder = FatBoostFeatureBuilder(
            min_df=1, word_max_features=20, char_max_features=30, embedding_model="fake",
        )
        result = builder.fit_transform(items, pairs, encoder=FakeEncoder())  # type: ignore[arg-type]
        embedding_columns = [name for name in builder.numeric_feature_names if name.startswith("embedding_")]
        self.assertEqual(len(embedding_columns), 5)
        self.assertTrue(np.isfinite(result[embedding_columns].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
