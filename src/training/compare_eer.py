"""Compare checkpoint verification EER using the same held-out image pairs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.dataset import FingerprintDataset
from src.retrieval.core import load_model
from src.training.eval import embed_dataset


def make_pairs(meta, count, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, label in enumerate(meta.class_id.astype(str)):
        groups[label].append(i)
    eligible = [v for v in groups.values() if len(v) >= 2]
    labels = meta.class_id.astype(str).to_numpy()
    genuine, impostor = set(), set()
    while len(genuine) < count:
        group = eligible[int(rng.integers(len(eligible)))]
        a, b = rng.choice(group, 2, replace=False)
        genuine.add(tuple(sorted((int(a), int(b)))))
    while len(impostor) < count:
        a, b = rng.choice(len(meta), 2, replace=False)
        if labels[a] != labels[b]:
            impostor.add(tuple(sorted((int(a), int(b)))))
    return [(a, b, 1) for a, b in sorted(genuine)] + [(a, b, 0) for a, b in sorted(impostor)]


def metrics(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    frr = 1 - tpr
    diff = fpr - frr
    right = int(np.flatnonzero(diff >= 0)[0])
    left = max(0, right - 1)
    weight = -diff[left] / (diff[right] - diff[left]) if diff[right] != diff[left] else 0.0
    eer = float(fpr[left] + weight * (fpr[right] - fpr[left]))
    # Report an achievable threshold separately; interpolated EER need not be attainable.
    nearest = int(np.argmin(np.abs(diff)))
    return {"eer": eer, "auc": float(roc_auc_score(labels, scores)),
            "nearest_threshold": float(thresholds[nearest]),
            "far_at_threshold": float(fpr[nearest]), "frr_at_threshold": float(frr[nearest]),
            "genuine_mean": float(scores[labels == 1].mean()),
            "impostor_mean": float(scores[labels == 0].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/processed/test/test.csv")
    ap.add_argument("--output", type=Path, default=Path("outputs/eer_comparison"))
    ap.add_argument("--pairs-per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(4)
    dataset = FingerprintDataset(args.csv, augment_images=False)
    pairs = make_pairs(dataset.df, args.pairs_per_class, args.seed)
    selected = sorted({i for a, b, _ in pairs for i in (a, b)})
    mapping = {row: i for i, row in enumerate(selected)}
    left = np.array([mapping[a] for a, _, _ in pairs]); right = np.array([mapping[b] for _, b, _ in pairs])
    labels = np.array([label for _, _, label in pairs])
    args.output.mkdir(parents=True, exist_ok=True)
    pair_data = pd.DataFrame(pairs, columns=["row_a", "row_b", "genuine"])
    pair_data["class_a"] = dataset.df.iloc[pair_data.row_a].class_id.to_numpy()
    pair_data["class_b"] = dataset.df.iloc[pair_data.row_b].class_id.to_numpy()
    pair_data.to_csv(args.output / "test_pairs.csv", index=False)
    checkpoints = sorted(Path("checkpoints").rglob("best*.pt"))
    summaries = []
    for checkpoint in checkpoints:
        name = "__".join(checkpoint.relative_to("checkpoints").with_suffix("").parts)
        print(f"Evaluating {name} ({len(selected)} unique test images)", flush=True)
        try:
            model, device, backbone, dim = load_model(str(checkpoint))
            vectors, _ = embed_dataset(model, dataset, selected, device, args.batch_size, log_every=10)
            scores = np.einsum("ij,ij->i", vectors[left], vectors[right])
            report = dict(model=name, checkpoint=str(checkpoint), backbone=backbone, dimension=dim,
                          seed=args.seed, genuine_pairs=int(labels.sum()), impostor_pairs=int((1-labels).sum()),
                          unique_images=len(selected), **metrics(scores, labels))
            pd.DataFrame({"row_a": pair_data.row_a, "row_b": pair_data.row_b,
                          "genuine": labels, "cosine": scores}).to_csv(args.output / (name + "_scores.csv"), index=False)
            (args.output / (name + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
            summaries.append(report)
            print(f"  EER {report['eer']:.2%}; AUC {report['auc']:.4f}", flush=True)
            del model, vectors
            if device.type == "cuda": torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
            summaries.append({"model": name, "checkpoint": str(checkpoint), "error": str(exc)})
    pd.DataFrame(summaries).to_csv(args.output / "comparison.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
