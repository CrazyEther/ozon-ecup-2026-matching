from __future__ import annotations

import unittest

import numpy as np

from src.ensemble import blend_scores


class EnsembleBlendTest(unittest.TestCase):
    def test_endpoints_return_individual_models(self) -> None:
        lora = np.array([0.1, 0.8, 0.4])
        fatboost = np.array([0.7, 0.2, 0.6])
        np.testing.assert_allclose(
            blend_scores(lora, fatboost, fatboost_weight=0.0, method="linear"), lora,
        )
        np.testing.assert_allclose(
            blend_scores(lora, fatboost, fatboost_weight=1.0, method="linear"), fatboost,
        )

    def test_rank_blend_is_per_group(self) -> None:
        lora = np.array([0.1, 0.9, 0.8, 0.2])
        fatboost = np.array([0.2, 0.8, 0.9, 0.1])
        groups = np.array(["a", "a", "b", "b"])
        actual = blend_scores(
            lora, fatboost, fatboost_weight=0.5, method="rank", groups=groups,
        )
        np.testing.assert_allclose(actual, np.array([0.5, 1.0, 1.0, 0.5]))

    def test_invalid_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            blend_scores(np.array([0.2]), np.array([0.3]), fatboost_weight=1.1)


if __name__ == "__main__":
    unittest.main()
