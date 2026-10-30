import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.retrieval.upload_embeddings import upload_export


class UploadExportsTests(unittest.TestCase):
    def test_upload_and_resume_preserve_vectors_and_image_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "resnet" / "test"
            folder.mkdir(parents=True)
            manifest = {"complete": True, "rows": 2, "embedding_dim": 2, "split": "test"}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            np.save(folder / "embeddings.npy", np.eye(2, dtype=np.float32))
            pd.DataFrame({"class_id": ["a", "b"]}).to_csv(folder / "metadata.csv", index=False)
            remote = Mock()
            remote.describe_index_stats.side_effect = lambda: SimpleNamespace(namespaces={
                json.loads((folder / "pinecone.json").read_text())["namespace"]:
                    SimpleNamespace(vector_count=2)})
            remote.fetch.return_value = SimpleNamespace(vectors={
                "0": SimpleNamespace(values=[1., 0.]), "1": SimpleNamespace(values=[0., 1.])})
            with patch("src.retrieval.upload_embeddings.connect", return_value=remote):
                upload_export(folder, "test-index", batch_size=1)
                self.assertEqual(remote.upsert.call_count, 2)
                self.assertEqual(remote.upsert.call_args.kwargs["vectors"][0]["metadata"]["class_id"], "b")
                upload_export(folder, "test-index", batch_size=1)
                self.assertEqual(remote.upsert.call_count, 2)
            self.assertTrue(json.loads((folder / "pinecone.json").read_text())["complete"])

    def test_incomplete_export_never_connects(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "manifest.json").write_text('{"complete": false}')
            with patch("src.retrieval.upload_embeddings.connect") as connect:
                with self.assertRaises(ValueError):
                    upload_export(folder, "test-index")
                connect.assert_not_called()
