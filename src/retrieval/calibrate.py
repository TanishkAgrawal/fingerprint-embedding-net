"""Calibrate a cosine decision threshold ONLY on the validation subjects.

This calibrates 1:1 verification using genuine pairs and sampled impostor pairs.
It does not measure open-set identification/rejection performance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from src.data.dataset import FingerprintDataset
from .core import load_model, read_split, embed_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-csv", default="data/processed/val/val.csv")
    ap.add_argument("--out", default="outputs/retrieval/densenet121/threshold.json")
    ap.add_argument("--target-far", type=float, default=0.001)
    ap.add_argument("--impostor-pairs", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if not 0 <= args.target_far < 1:
        raise ValueError("target-far must be in [0, 1)")
    df = read_split(args.val_csv)
    dataset = FingerprintDataset(args.val_csv, augment_images=False)
    if len(dataset) != len(df):
        raise ValueError("Dataset/CSV length mismatch")
    model, dev, backbone, dim = load_model(args.checkpoint)
    vec = embed_rows(model, dev, dataset, list(range(len(dataset))), args.batch_size)
    labels = df["class_id"].to_numpy()
    genuine_parts = []
    for _, group in df.groupby("class_id"):
        ids = group.index.to_numpy(dtype=int)
        if len(ids) < 2:
            continue
        sim = vec[ids] @ vec[ids].T
        a, b = np.triu_indices(len(ids), k=1)
        genuine_parts.append(sim[a, b])
    if not genuine_parts:
        raise ValueError("No genuine pairs in validation CSV")
    genuine = np.concatenate(genuine_parts)
    rng = np.random.default_rng(42)
    imp = []
    remaining = args.impostor_pairs
    while remaining > 0:
        count = min(max(remaining * 2, 2048), 200000)
        i = rng.integers(0, len(vec), size=count)
        j = rng.integers(0, len(vec), size=count)
        valid = labels[i] != labels[j]
        i, j = i[valid][:remaining], j[valid][:remaining]
        imp.append(np.sum(vec[i] * vec[j], axis=1))
        remaining -= len(i)
    impostor = np.concatenate(imp)
    # Conservative empirical threshold: number accepted impostors <= floor(target FAR * N).
    descending = np.sort(impostor)[::-1]
    allowed = int(np.floor(args.target_far * len(descending)))
    if allowed >= len(descending):
        raise ValueError("Unusable FAR target")
    threshold = float(np.nextafter(descending[allowed], np.inf))
    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(genuine < threshold))
    result = {"checkpoint": args.checkpoint, "val_csv": args.val_csv,
              "backbone": backbone, "embedding_dim": dim,
              "target_far": args.target_far, "threshold": threshold,
              "empirical_val_far": far, "empirical_val_frr": frr,
              "genuine_pairs": int(len(genuine)), "sampled_impostor_pairs": int(len(impostor)),
              "warning": "Threshold from validation verification pairs; evaluate open-set rejection separately."}
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved threshold to {dest}")

if __name__ == "__main__":
    main()
