import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.retrieval.pinecone_store import PineconeGallery, upload


class PineconeGalleryTests(unittest.TestCase):
    def setUp(self):
        self.remote = Mock()
        self.gallery = pd.DataFrame({"csv_row": [15, 3], "class_id": ["a", "b"]})
        self.store = PineconeGallery(self.remote, "isolated-build", self.gallery, 2)

    def test_remote_ids_map_to_gallery_positions(self):
        self.remote.query.return_value = SimpleNamespace(matches=[
            SimpleNamespace(id="3", score=0.9), SimpleNamespace(id="15", score=0.2)])
        scores, ids = self.store.search(np.array([[1., 0.]]), 2)
        self.assertEqual(ids.tolist(), [[1, 0]])
        self.assertAlmostEqual(float(scores[0, 0]), 0.9)
        self.assertEqual(self.remote.query.call_args.kwargs["namespace"], "isolated-build")

    def test_incomplete_or_foreign_results_fail_instead_of_mislabeling(self):
        for matches in ([], [SimpleNamespace(id="unknown", score=1.)]):
            self.remote.query.return_value = SimpleNamespace(matches=matches)
            with self.assertRaises(RuntimeError):
                self.store.search([[1., 0.]], 1)

    def test_invalid_queries_do_not_reach_remote(self):
        for vectors, k in (([[1.]], 1), ([[float("nan"), 0.]], 1), ([[1., 0.]], 0)):
            with self.assertRaises(ValueError):
                self.store.search(vectors, k)
        self.remote.query.assert_not_called()

    def test_upload_preserves_ids_metadata_and_namespace(self):
        self.remote.describe_index_stats.return_value = SimpleNamespace(
            namespaces={"isolated-build": SimpleNamespace(vector_count=2)})
        upload(self.remote, "isolated-build", np.eye(2), self.gallery)
        args = self.remote.upsert.call_args.kwargs
        self.assertEqual(args["namespace"], "isolated-build")
        self.assertEqual(args["vectors"][1], {
            "id": "3", "values": [0., 1.], "metadata": {"class_id": "b", "csv_row": 3}})


if __name__ == "__main__":
    unittest.main()
