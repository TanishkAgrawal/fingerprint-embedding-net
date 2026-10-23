"""Held-out identification evaluation, with one enrolled image per finger."""
import argparse
import numpy as np
from src.data.dataset import FingerprintDataset
from .core import load_model, embed_rows, load_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="outputs/retrieval/densenet121")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    index, gallery, queries, manifest = load_index(args.index_dir)
    model, device, _, dim = load_model(manifest["checkpoint"])
    if dim != index.d:
        raise RuntimeError("Model and index embedding dimensions differ")
    dataset = FingerprintDataset(manifest["test_csv"], augment_images=False)
    vectors = embed_rows(model, device, dataset, queries["csv_row"].astype(int).tolist(), args.batch_size)
    k = min(args.top_k, index.ntotal)
    scores, ids = index.search(vectors, k)
    enrolled = gallery["class_id"].astype(str).to_numpy()
    expected = queries["class_id"].astype(str).to_numpy()
    found = (enrolled[ids] == expected[:, None])
    rank1 = found[:, 0].mean()
    recall_k = found.any(axis=1).mean()
    print(f"Gallery identities : {len(gallery)}")
    print(f"Query images       : {len(queries)}")
    print(f"Rank-1 accuracy    : {rank1:.4%}")
    print(f"Recall@{k:<2}        : {recall_k:.4%}")
    print("Note: closed-set identification; no-match threshold and open-set rejection not evaluated.")

if __name__ == "__main__":
    main()
