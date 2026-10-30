import unittest

import numpy as np
import pandas as pd

from src.training.compare_eer import make_pairs, metrics


class ComparisonTests(unittest.TestCase):
    def test_pairs_are_unique_distinct_and_reproducible(self):
        metadata = pd.DataFrame({"class_id": ["a", "a", "a", "b", "b", "b"]})
        pairs = make_pairs(metadata, 4, 42)
        self.assertEqual(pairs, make_pairs(metadata, 4, 42))
        self.assertEqual(len(set((a, b) for a, b, _ in pairs)), 8)
        for a, b, label in pairs:
            self.assertNotEqual(a, b)
            self.assertEqual(int(metadata.class_id[a] == metadata.class_id[b]), label)

    def test_eer_and_auc_for_perfect_separation(self):
        result = metrics(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]))
        self.assertEqual(result["eer"], 0.0)
        self.assertEqual(result["auc"], 1.0)
