"""
Complete test-set evaluation for a trained fingerprint embedding model.

Reports:
  - Verification: EER, ROC-AUC, genuine/impostor cosine stats
  - Same metrics split by alteration difficulty (easy / medium / hard)
  - Identification: rank-1 / rank-5 / rank-10 (gallery = real, probe = altered)

Saves JSON + ROC / histogram / CMC plots under outputs/results/.

    python -m src.training.full_eval --config src/training/config.yaml --which latest
    python -m src.training.full_eval --config src/training/config.yaml --checkpoint checkpoints/resnet18_arcface/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.dataset import FingerprintDataset
from src.models.backbone import FingerprintEmbeddingNet
from src.training.eval import (
    _log,
    embed_dataset,
    equal_error_rate,
    load_config,
    pair_scores,
    resolve_checkpoint,
    resolve_device,
)
import torch


def _roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    s = np.concatenate([genuine, impostor])
    return float(roc_auc_score(y, s))


def _verify_block(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    if genuine.size == 0 or impostor.size == 0:
        return {"n_genuine": int(genuine.size), "n_impostor": int(impostor.size),
                "eer": None, "roc_auc": None, "gap": None,
                "genuine_mean": None, "impostor_mean": None}
    return {
        "n_genuine": int(len(genuine)),
        "n_impostor": int(len(impostor)),
        "eer": equal_error_rate(genuine, impostor),
        "roc_auc": _roc_auc(genuine, impostor),
        "gap": float(genuine.mean() - impostor.mean()),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
    }


def genuine_by_column(embeddings: np.ndarray, meta, column: str, seed: int):
    """
    Genuine = real vs non-real of the same class_id, grouped by the non-real
    image's column value (difficulty or alteration). Impostors are shared.
    """
    rng = np.random.default_rng(seed)
    by_id = defaultdict(lambda: {"real": [], "other": []})
    for i, row in meta.iterrows():
        bucket = "real" if row["alteration"] == "real" else "other"
        by_id[row["class_id"]][bucket].append(i)

    grouped = defaultdict(list)
    for parts in by_id.values():
        for ri in parts["real"]:
            for aj in parts["other"]:
                key = str(meta.iloc[aj][column])
                grouped[key].append(float(np.dot(embeddings[ri], embeddings[aj])))

    impostor = []
    ids = [cid for cid, p in by_id.items() if p["real"]]
    n_target = min(sum(len(v) for v in grouped.values()) * 5, 200_000)
    n_try = 0
    while len(impostor) < n_target and n_try < n_target * 20 and len(ids) >= 2:
        n_try += 1
        a, b = rng.choice(len(ids), size=2, replace=False)
        real_pool = by_id[ids[a]]["real"]
        other_pool = by_id[ids[b]]["other"] or by_id[ids[b]]["real"]
        if not real_pool or not other_pool:
            continue
        i = int(rng.choice(real_pool))
        j = int(rng.choice(other_pool))
        impostor.append(float(np.dot(embeddings[i], embeddings[j])))
    impostor = np.asarray(impostor, dtype=np.float32)

    out = {}
    for key, scores in sorted(grouped.items()):
        out[key] = _verify_block(np.asarray(scores, dtype=np.float32), impostor)
    return out


def identification(embeddings: np.ndarray, meta, ks=(1, 5, 10)) -> dict:
    """Gallery = real prints; probe = altered prints. Rank of true class_id."""
    gallery_idx = meta.index[meta["alteration"] == "real"].to_numpy()
    probe_idx = meta.index[meta["alteration"] != "real"].to_numpy()
    if len(gallery_idx) == 0 or len(probe_idx) == 0:
        return {"n_gallery": int(len(gallery_idx)), "n_probe": int(len(probe_idx))}

    G = embeddings[gallery_idx]
    P = embeddings[probe_idx]
    gallery_ids = meta.iloc[gallery_idx]["class_id"].to_numpy()
    probe_ids = meta.iloc[probe_idx]["class_id"].to_numpy()
    probe_diff = meta.iloc[probe_idx]["difficulty"].to_numpy()

    sims = P @ G.T
    order = np.argsort(-sims, axis=1)
    ranked_ids = gallery_ids[order]

    ranks = np.empty(len(probe_idx), dtype=np.int32)
    for i in range(len(probe_idx)):
        hits = np.where(ranked_ids[i] == probe_ids[i])[0]
        ranks[i] = int(hits[0] + 1) if len(hits) else int(len(gallery_ids) + 1)

    def cmc_at(rank_arr, k):
        return float(np.mean(rank_arr <= k))

    result = {
        "n_gallery": int(len(gallery_idx)),
        "n_probe": int(len(probe_idx)),
        "mean_rank": float(ranks.mean()),
    }
    for k in ks:
        result[f"rank_{k}"] = cmc_at(ranks, k)

    by_diff = {}
    for diff in sorted(set(probe_diff.tolist())):
        mask = probe_diff == diff
        block = {"n_probe": int(mask.sum())}
        for k in ks:
            block[f"rank_{k}"] = cmc_at(ranks[mask], k)
        by_diff[str(diff)] = block
    result["by_difficulty"] = by_diff
    result["ranks"] = ranks
    return result


def save_plots(out_dir: str, genuine, impostor, ident: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(impostor, bins=40, alpha=0.65, label="impostor")
    ax.hist(genuine, bins=40, alpha=0.65, label="genuine")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    ax.set_title("Verification scores (full test)")
    ax.legend()
    fig.tight_layout()
    paths["histogram"] = os.path.join(out_dir, "histogram.png")
    fig.savefig(paths["histogram"], dpi=140)
    plt.close(fig)

    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    s = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, s):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False accept rate")
    ax.set_ylabel("True accept rate")
    ax.set_title("ROC (verification)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    paths["roc"] = os.path.join(out_dir, "roc.png")
    fig.savefig(paths["roc"], dpi=140)
    plt.close(fig)

    ranks = ident.get("ranks")
    if ranks is not None:
        ks = np.arange(1, min(21, ident["n_gallery"] + 1))
        cmc = [float(np.mean(ranks <= k)) for k in ks]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(ks, cmc, marker="o")
        ax.set_xlabel("Rank k")
        ax.set_ylabel("Identification accuracy")
        ax.set_title("CMC curve (gallery=real, probe=altered)")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        paths["cmc"] = os.path.join(out_dir, "cmc.png")
        fig.savefig(paths["cmc"], dpi=140)
        plt.close(fig)

    return paths


def _print_verify(title: str, block: dict) -> None:
    _log(f"  {title}")
    if block.get("eer") is None:
        _log("    (empty)")
        return
    _log(f"    pairs  genuine={block['n_genuine']:,}  impostor={block['n_impostor']:,}")
    _log(f"    EER={block['eer']:.4f}  ROC-AUC={block['roc_auc']:.4f}  "
         f"gap={block['gap']:.4f}")
    _log(f"    cosine genuine={block['genuine_mean']:.4f}  "
         f"impostor={block['impostor_mean']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Full fingerprint test evaluation")
    parser.add_argument("--config", default="src/training/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--which", choices=["best", "latest"], default="latest")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    ckpt_path = resolve_checkpoint(cfg, which=args.which, explicit=args.checkpoint)
    csv_path = args.csv or cfg["data"].get("test_csv", "data/processed/test/test.csv")
    if os.path.isdir(ckpt_path):
        raise SystemExit(
            f"Checkpoint path is a FOLDER, not a weight file:\n  {ckpt_path}\n"
            "A .pt file is already a zip. Do not Extract All on it.\n"
            "Copy the original .pt from the other machine without unzipping, "
            "or keep the repacked file in checkpoints/<run>/latest.pt"
        )
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    if os.path.isdir(ckpt_path):
        raise SystemExit(
            f"Checkpoint is a folder, not a file: {ckpt_path}\n"
            "A .pt file is already a zip. Do not Extract All on it.\n"
            "Copy the raw .pt from the other machine, or re-zip the inner "
            "folder (the one that contains data.pkl) back into a .pt file."
        )
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", cfg["model"]["backbone"])
    loss_type = ckpt.get("loss_type", cfg["loss"]["type"])
    embedding_dim = cfg["model"]["embedding_dim"]
    epoch = ckpt.get("epoch", "?")

    _log(f"\n{'=' * 60}")
    _log("  Full test evaluation")
    _log(f"  Checkpoint : {ckpt_path}")
    _log(f"  Epoch      : {epoch}   {backbone} / {loss_type} / dim={embedding_dim}")
    _log(f"  Device     : {device}")
    _log(f"{'=' * 60}\n")

    model = FingerprintEmbeddingNet(backbone, embedding_dim, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    dataset = FingerprintDataset(csv_path, augment_images=False)
    indices = list(range(len(dataset)))
    _log(f"  Embedding {len(dataset)} test images...")
    embeddings, class_ids = embed_dataset(
        model, dataset, indices, device, args.batch_size
    )
    meta = dataset.df.iloc[indices].reset_index(drop=True)
    assert len(meta) == len(embeddings)

    genuine, impostor = pair_scores(embeddings, class_ids, args.seed)
    overall = _verify_block(genuine, impostor)
    by_diff = genuine_by_column(embeddings, meta, "difficulty", args.seed)
    by_alt = genuine_by_column(embeddings, meta, "alteration", args.seed)
    ident = identification(embeddings, meta)

    _log("\n  -- Verification (all pairs) --")
    _print_verify("overall", overall)
    _log("\n  -- Verification by probe difficulty (real vs altered, same finger) --")
    for k, block in by_diff.items():
        if k == "none":
            continue
        _print_verify(k, block)
    _log("\n  -- Verification by alteration type --")
    for k, block in by_alt.items():
        if k == "real":
            continue
        _print_verify(k, block)

    _log("\n  -- Identification (gallery=real, probe=altered) --")
    _log(f"    gallery={ident['n_gallery']}  probe={ident['n_probe']}")
    _log(f"    rank-1={ident['rank_1']:.4f}  rank-5={ident['rank_5']:.4f}  "
         f"rank-10={ident['rank_10']:.4f}  mean_rank={ident['mean_rank']:.2f}")
    for diff, block in ident.get("by_difficulty", {}).items():
        _log(f"    {diff:8s}  n={block['n_probe']:<5}  "
             f"rank-1={block['rank_1']:.4f}  rank-5={block['rank_5']:.4f}")

    tag = args.experiment or f"{backbone}_{loss_type}_e{epoch}"

    base_dir = cfg["output"].get(
        "results_dir", "outputs/results"
    )
    if args.experiment:
        out_dir = os.path.join(base_dir, "tune", tag)
    else:
        out_dir = os.path.join(base_dir, tag)
    plots = save_plots(out_dir, genuine, impostor, ident)

    report = {
        "checkpoint": ckpt_path,
        "epoch": epoch,
        "backbone": backbone,
        "loss_type": loss_type,
        "embedding_dim": embedding_dim,
        "n_images": int(len(meta)),
        "n_identities": int(meta["class_id"].nunique()),
        "verification_overall": overall,
        "verification_by_difficulty": by_diff,
        "verification_by_alteration": {k: v for k, v in by_alt.items() if k != "real"},
        "identification": {k: v for k, v in ident.items() if k != "ranks"},
        "plots": plots,
    }
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    np.savez(os.path.join(out_dir, "scores.npz"), genuine=genuine, impostor=impostor)
    _log(f"\n  Report -> {report_path}")
    for name, path in plots.items():
        _log(f"  {name:10s} -> {path}")
    _log("")


if __name__ == "__main__":
    main()
