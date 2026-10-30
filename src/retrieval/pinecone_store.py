"""Pinecone storage for locally computed fingerprint embeddings."""
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


def load_settings():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def connect(name, dimension, create=False):
    from pinecone import Pinecone, ServerlessSpec

    load_settings()
    key = os.environ.get("PINECONE_API_KEY")
    if not key:
        raise ValueError("Set PINECONE_API_KEY in your environment before using retrieval")
    pc = Pinecone(api_key=key)
    if not pc.has_index(name):
        if not create:
            raise ValueError(f"Pinecone index {name!r} does not exist; run build_index first")
        pc.create_index(name=name, dimension=dimension, metric="cosine",
                        spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                            region=os.getenv("PINECONE_REGION", "us-east-1")))
    deadline = time.monotonic() + 180
    while True:
        description = pc.describe_index(name)
        if description.dimension != dimension or description.metric != "cosine":
            raise ValueError("Pinecone index must use cosine and match the model embedding dimension")
        if description.status["ready"]:
            return pc.Index(host=description.host)
        if time.monotonic() >= deadline:
            raise TimeoutError("Pinecone index did not become ready within 180 seconds")
        time.sleep(2)


class PineconeGallery:
    """Translate remote vector IDs to the local gallery's positional row IDs."""

    def __init__(self, index, namespace, gallery, dimension):
        self.index = index
        self.namespace = namespace
        self.d = dimension
        self.ntotal = len(gallery)
        self.positions = {str(int(row)): i for i, row in enumerate(gallery["csv_row"])}
        if len(self.positions) != self.ntotal:
            raise ValueError("Gallery contains duplicate csv_row IDs")

    def search(self, vectors, k):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.d or not np.isfinite(vectors).all():
            raise ValueError("Query vectors must be finite and match the index dimension")
        if not 1 <= k <= min(self.ntotal, 10000):
            raise ValueError("top-k must be positive and no larger than the gallery or 10000")
        scores, positions = [], []
        for vector in vectors:
            response = self.index.query(namespace=self.namespace, vector=vector.tolist(),
                                        top_k=k, include_values=False, include_metadata=False)
            matches = response.matches
            if len(matches) != k or any(m.id not in self.positions for m in matches):
                raise RuntimeError("Pinecone results do not match the saved gallery; wait for indexing or rebuild")
            scores.append([m.score for m in matches])
            positions.append([self.positions[m.id] for m in matches])
        return np.asarray(scores, dtype=np.float32), np.asarray(positions, dtype=np.int64)


def upload(index, namespace, vectors, gallery, timeout=180):
    for start in range(0, len(gallery), 100):
        records = [{"id": str(int(gallery.iloc[i]["csv_row"])),
                    "values": vectors[i].tolist(),
                    "metadata": {"class_id": str(gallery.iloc[i]["class_id"]),
                                 "csv_row": int(gallery.iloc[i]["csv_row"])}}
                   for i in range(start, min(start + 100, len(gallery)))]
        index.upsert(vectors=records, namespace=namespace)
    deadline = time.monotonic() + timeout
    while True:
        stats = index.describe_index_stats()
        entry = stats.namespaces.get(namespace)
        if entry is not None and entry.vector_count == len(gallery):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Pinecone ingestion not visible yet; local manifest was not updated")
        time.sleep(2)
