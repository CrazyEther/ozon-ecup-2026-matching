from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from src.fatboost import FatBoostFeatureBuilder
from train_catboost import make_pool


class FatBoostTrainingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        products = []
        for index in range(40):
            category = "phones" if index < 20 else "drinks"
            products.append({
                "id": index,
                "name": f"common product brand model A{index % 10} {500 + index % 2 * 500} ml",
                "attributes": '{"brand":"common","pack":"2 шт"}',
                "category": category,
            })
        cls.items = pd.DataFrame(products)
        pairs = []
        for category_start in (0, 20):
            for offset in range(20):
                pairs.append({
                    "id1": category_start + offset,
                    "id2": category_start + ((offset + (0 if offset % 2 else 1)) % 20),
                    "target": float(offset % 2),
                    "category": "phones" if category_start == 0 else "drinks",
                })
        cls.frame = pd.DataFrame(pairs)
        cls.builder = FatBoostFeatureBuilder(min_df=1, word_max_features=200, char_max_features=300)
        cls.features = cls.builder.fit_transform(cls.items, cls.frame)

    def test_classifier_accepts_native_text(self) -> None:
        indices = self.frame.index
        pool, _ = make_pool(self.features, self.frame, indices, self.builder, False)
        model = CatBoostClassifier(
            iterations=2, depth=2, loss_function="Logloss", verbose=False,
            allow_writing_files=False, thread_count=2,
        )
        model.fit(pool)
        expected = model.predict_proba(pool)
        self.assertEqual(len(expected), len(indices))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.cbm"
            model.save_model(path)
            restored = CatBoostClassifier()
            restored.load_model(path)
            actual = restored.predict_proba(Pool(
                self.features,
                cat_features=self.builder.cat_feature_names,
                text_features=self.builder.text_feature_names,
            ))
        np.testing.assert_allclose(expected, actual, rtol=1e-7)

    def test_ranker_accepts_map_groups(self) -> None:
        indices = self.frame.index
        pool, _ = make_pool(self.features, self.frame, indices, self.builder, True)
        model = CatBoostRanker(
            iterations=2, depth=2, loss_function="YetiRank:mode=MAP", verbose=False,
            allow_writing_files=False, thread_count=2,
        )
        model.fit(pool)
        self.assertTrue(np.isfinite(model.predict(pool)).all())

    def test_feature_builder_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.joblib"
            joblib.dump(self.builder, path)
            loaded: FatBoostFeatureBuilder = joblib.load(path)
            restored = loaded.transform(self.items, self.frame)
        np.testing.assert_allclose(
            restored[loaded.numeric_feature_names],
            self.features[self.builder.numeric_feature_names],
            rtol=1e-6,
        )
        self.assertEqual(restored[loaded.text_feature_names].to_dict(),
                         self.features[self.builder.text_feature_names].to_dict())


if __name__ == "__main__":
    unittest.main()
