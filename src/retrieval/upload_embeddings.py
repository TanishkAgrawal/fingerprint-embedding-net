"""Persist completed embedding exports in Pinecone; safely resume failed uploads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .export_embeddings import save_state
from .pinecone_store import connect, load_settings


def upload_export(directory, index_name, batch_size=100):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"Export incomplete: {directory}")
    vectors = np.load(directory / "embeddings.npy", mmap_mode="r")
    metadata = pd.read_csv(directory / "metadata.csv").fillna("")
    count, dim = vectors.shape
    if count != manifest["rows"] or count != len(metadata) or dim != manifest["embedding_dim"]:
        raise ValueError("Export metadata/vector dimensions do not agree")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
    namespace = f"all-images-{directory.parent.name[:60]}-{manifest['split']}-{digest}"
    remote = connect(index_name, dim, create=True)
    state_path = directory / "pinecone.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    identity = dict(index=index_name, namespace=namespace, rows=count, dimension=dim)
    if state and any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("Saved upload destination differs; use a separate export directory")
    start = int(state.get("uploaded_rows", 0))
    if not 0 <= start <= count:
        raise ValueError("Invalid saved upload progress")
    state = dict(identity, uploaded_rows=start, complete=False)
    save_state(state_path, state)
    def send_batch(offset):
        end = min(offset + batch_size, count)
        block = np.asarray(vectors[offset:end])
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-4):
            raise ValueError("Invalid embedding values")
        records = []
        for i in range(offset, end):
            row = metadata.iloc[i]
            info = {"embedding_row": i, "split": manifest["split"],
                    "model": directory.parent.name}
            for field in ("class_id", "filepath", "processed_path", "difficulty", "alteration"):
                if field in metadata:
                    info[field] = str(row[field])
            records.append({"id": str(i), "values": vectors[i].tolist(), "metadata": info})
        remote.upsert(namespace=namespace, vectors=records)
        return end

    # map yields in input order: progress never skips a failed earlier batch.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for end in pool.map(send_batch, range(start, count, batch_size)):
            state["uploaded_rows"] = end
            save_state(state_path, state)
            if end % 1000 == 0 or end == count:
                print(f"Uploaded {directory.parent.name}/{manifest['split']}: {end}/{count}", flush=True)
    deadline = time.monotonic() + 300
    while True:
        stats = remote.describe_index_stats().namespaces.get(namespace)
        if stats and stats.vector_count == count:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Upload acknowledged but count not visible yet; rerun to verify")
        time.sleep(2)
    sample_ids = sorted({"0", str(count - 1)})
    fetched = remote.fetch(ids=sample_ids, namespace=namespace).vectors
    for key in sample_ids:
        if key not in fetched or not np.allclose(fetched[key].values, vectors[int(key)], atol=1e-6):
            raise RuntimeError("Remote embedding read-back verification failed")
    save_state(state_path, dict(state, complete=True, verified_remote_count=count,
                               verified_sample_ids=sample_ids))
    print(f"Verified {count} persisted vectors in {index_name}/{namespace}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True, help="Model export directory containing split folders")
    parser.add_argument("--index", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100:
        parser.error("batch-size must be between 1 and 100")
    load_settings()
    manifests = sorted(args.export_dir.glob("*/manifest.json"))
    if not manifests:
        parser.error("No exported splits found")
    for manifest in manifests:
        upload_export(manifest.parent, args.index or os.getenv("PINECONE_INDEX", "fingerprint-embeddings"), args.batch_size)


if __name__ == "__main__":
    main()
