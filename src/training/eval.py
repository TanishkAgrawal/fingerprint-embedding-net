"""
Embedding evaluation for fingerprint verification.

Training never sees test subjects, so we do not classify into the ArcFace
4200-class head. We embed images and score pairs:

  genuine  — same class_id  (same finger of the same person)
  impostor — different class_id

Cosine similarity is the match score (embeddings are already L2-normalised).

Smoke check (few hundred images, a few identities):
    python -m src.training.eval --config src/training/config.yaml

Full test set:
    python -m src.training.eval --config src/training/config.yaml --full
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FingerprintDataset
from src.models.backbone import FingerprintEmbeddingNet


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    preference = cfg.get("training", {}).get("device", "cuda").lower().strip()
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        idx = cfg.get("training", {}).get("cuda_device", 0)
        return torch.device(f"cuda:{idx}")
    if preference == "auto":
        return torch.device("cpu")
    _log("ERROR: CUDA requested but torch.cuda.is_available() is False.")
    _log(f"  Python  : {sys.executable}")
    _log(f"  PyTorch : {torch.__version__}")
    sys.exit(1)


def default_checkpoint(cfg: dict) -> str:
    backbone = cfg["model"]["backbone"]
    loss_type = cfg["loss"]["type"]
    return os.path.join(
        cfg["output"]["checkpoint_dir"], f"{backbone}_{loss_type}", "latest.pt"
    )


def subset_indices(dataset: FingerprintDataset, max_images: int, seed: int) -> list[int]:
    """
    Keep whole identities so genuine pairs exist. Add identities until
    we reach (or slightly exceed) max_images.
    """
    rng = np.random.default_rng(seed)
    by_id = defaultdict(list)
    for i, class_id in enumerate(dataset.df["class_id"]):
        by_id[class_id].append(i)

    # Need at least 2 images of the same finger for a genuine pair.
    eligible = [cid for cid, idxs in by_id.items() if len(idxs) >= 2]
    rng.shuffle(eligible)

    chosen = []
    n_ids = 0
    for cid in eligible:
        chosen.extend(by_id[cid])
        n_ids += 1
        if len(chosen) >= max_images:
            break

    if n_ids < 2:
        raise RuntimeError(
            "Need at least two finger identities with 2+ images each. "
            "Increase --max-images or use --full."
        )
    return chosen


@torch.no_grad()
def embed_dataset(model, dataset, indices, device, batch_size: int,
                  log_every: int | None = 20):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    chunks = []
    pin = device.type == "cuda"
    n_total = len(loader)
    for step, (imgs, _) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=pin)
        chunks.append(model(imgs).cpu())
        if log_every and (step == 1 or step % log_every == 0 or step == n_total):
            _log(f"    embed {step}/{n_total}")
    embeddings = torch.cat(chunks, dim=0).numpy()
    class_ids = dataset.df.iloc[indices]["class_id"].to_numpy()
    return embeddings, class_ids


@torch.no_grad()
def verification_metrics(model, dataset, device, batch_size: int = 64,
                         seed: int = 42, quiet: bool = True) -> dict:
    """Genuine vs impostor cosine stats using class_id (not ArcFace labels)."""
    indices = list(range(len(dataset)))
    embeddings, class_ids = embed_dataset(
        model, dataset, indices, device, batch_size,
        log_every=None if quiet else 20,
    )
    genuine, impostor = pair_scores(embeddings, class_ids, seed)
    return {
        "eer": equal_error_rate(genuine, impostor),
        "gap": float(genuine.mean() - impostor.mean()),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
    }


def pair_scores(embeddings: np.ndarray, class_ids: np.ndarray, seed: int,
                impostor_ratio: int = 5):
    """All genuine pairs in the subset + sampled impostor pairs."""
    rng = np.random.default_rng(seed)
    by_id = defaultdict(list)
    for i, cid in enumerate(class_ids):
        by_id[cid].append(i)

    genuine = []
    for idxs in by_id.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                genuine.append(float(np.dot(embeddings[i], embeddings[j])))

    genuine = np.asarray(genuine, dtype=np.float32)
    if genuine.size == 0:
        raise RuntimeError("No genuine pairs found (need 2+ images per class_id).")

    ids = list(by_id.keys())
    n_impostor = min(len(genuine) * impostor_ratio, 200_000)
    impostor = []
    n_try = 0
    max_try = n_impostor * 20
    while len(impostor) < n_impostor and n_try < max_try:
        n_try += 1
        a, b = rng.choice(len(ids), size=2, replace=False)
        i = int(rng.choice(by_id[ids[a]]))
        j = int(rng.choice(by_id[ids[b]]))
        impostor.append(float(np.dot(embeddings[i], embeddings[j])))

    impostor = np.asarray(impostor, dtype=np.float32)
    return genuine, impostor


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """
    FAR = fraction of impostors scoring at/above t
    FRR = fraction of genuines scoring below t
    EER is the operating point where FAR ~ FRR.
    """
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([
        np.ones(len(genuine), dtype=np.float32),
        np.zeros(len(impostor), dtype=np.float32),
    ])
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    far = fp / max(fp[-1], 1.0)
    frr = 1.0 - tp / max(tp[-1], 1.0)
    idx = int(np.argmin(np.abs(far - frr)))
    return float(0.5 * (far[idx] + frr[idx]))


def summarize(genuine: np.ndarray, impostor: np.ndarray) -> None:
    eer = equal_error_rate(genuine, impostor)
    gap = genuine.mean() - impostor.mean()
    _log("")
    _log(f"  Genuine pairs  : {len(genuine):,}")
    _log(f"    cosine mean  : {genuine.mean():.4f}   std={genuine.std():.4f}")
    _log(f"    min / max    : {genuine.min():.4f} / {genuine.max():.4f}")
    _log(f"  Impostor pairs : {len(impostor):,}")
    _log(f"    cosine mean  : {impostor.mean():.4f}   std={impostor.std():.4f}")
    _log(f"    min / max    : {impostor.min():.4f} / {impostor.max():.4f}")
    _log(f"  Mean gap       : {gap:.4f}   (genuine - impostor; >0 is good)")
    _log(f"  EER            : {eer:.4f}   (lower is better; 0.5 ~ random)")
    _log("")
    if eer < 0.10:
        _log("  Sanity: strong same-finger vs different-finger separation.")
    elif eer < 0.25:
        _log("  Sanity: real separation, but histograms still overlap.")
    else:
        _log("  Sanity: weak separation — embeddings not discriminative enough yet.")


def main():
    parser = argparse.ArgumentParser(description="Fingerprint embedding verification eval")
    parser.add_argument("--config", default="src/training/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit .pt path (overrides --which)")
    parser.add_argument("--which", choices=["best", "latest"], default="best",
                        help="Which saved run to load when --checkpoint is omitted")
    parser.add_argument("--csv", default=None,
                        help="Override test CSV path from config")
    parser.add_argument("--full", action="store_true",
                        help="Evaluate the entire CSV (skip the small subset)")
    parser.add_argument("--max-images", type=int, default=400,
                        help="Approx. image cap for the smoke subset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    ckpt_path = args.checkpoint or default_checkpoint(cfg)
    if args.checkpoint is None and args.which == "latest":
        ckpt_path = ckpt_path.replace("best.pt", "latest.pt")
    csv_path = args.csv or cfg["data"].get("test_csv", "data/processed/test/test.csv")

    if not os.path.exists(ckpt_path):
        _log(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not os.path.exists(csv_path):
        _log(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    _log(f"\n{'=' * 60}")
    _log(f"  Eval      : verification (cosine on embeddings)")
    _log(f"  Device    : {device}")
    _log(f"  Checkpoint: {ckpt_path}")
    _log(f"  CSV       : {csv_path}")
    _log(f"{'=' * 60}\n")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", cfg["model"]["backbone"])
    embedding_dim = cfg["model"]["embedding_dim"]
    _log(f"  Backbone  : {backbone}   embedding_dim={embedding_dim}")
    epoch = ckpt.get("epoch", "?")
    extra = f", val_eer={ckpt['val_eer']:.4f}" if "val_eer" in ckpt else ""
    _log(f"  Trained to epoch {epoch}  "
         f"(val_loss={ckpt.get('val_loss', float('nan')):.4f}{extra})")
    if args.checkpoint is None and args.which == "best" and isinstance(epoch, int) and epoch <= 2:
        latest = default_checkpoint(cfg).replace("best.pt", "latest.pt")
        if os.path.exists(latest):
            _log("  NOTE     : best.pt is lowest ArcFace val-loss, often epoch 1")
            _log("             on this open-set split. Last-epoch weights:")
            _log("             python -m src.training.eval --config src/training/config.yaml --full --which latest")

    model = FingerprintEmbeddingNet(backbone, embedding_dim, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    dataset = FingerprintDataset(csv_path, augment_images=False)
    if args.full:
        indices = list(range(len(dataset)))
        mode = "full"
    else:
        indices = subset_indices(dataset, args.max_images, args.seed)
        mode = "smoke"

    n_ids = dataset.df.iloc[indices]["class_id"].nunique()
    _log(f"  Mode      : {mode}   images={len(indices)}   identities={n_ids}\n")
    _log("  Embedding...")
    embeddings, class_ids = embed_dataset(
        model, dataset, indices, device, args.batch_size
    )
    _log(f"  Embedding shape: {embeddings.shape}  "
         f"(expected unit-norm, dim={embedding_dim})")
    norms = np.linalg.norm(embeddings, axis=1)
    _log(f"  Embedding L2   : mean={norms.mean():.4f}  "
         f"min={norms.min():.4f}  max={norms.max():.4f}")

    genuine, impostor = pair_scores(embeddings, class_ids, args.seed)
    summarize(genuine, impostor)

    out_dir = cfg["output"].get("log_dir", "outputs/logs")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{backbone}_{ckpt.get('loss_type', cfg['loss']['type'])}_{mode}_e{epoch}"
    out_path = os.path.join(out_dir, f"eval_{tag}.npz")
    np.savez(out_path, genuine=genuine, impostor=impostor)
    _log(f"  Scores saved -> {out_path}\n")


if __name__ == "__main__":
    main()
