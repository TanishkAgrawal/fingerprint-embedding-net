"""Export all fingerprint embeddings from saved, trained projection heads.

Run from the repository root: python -m src.retrieval.export_embeddings
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FingerprintDataset
from .core import load_model


def signature(checkpoint, csv):
    stat = checkpoint.stat()
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_size": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
            "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest()}


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


@torch.inference_mode()
def export_split(model, device, dim, checkpoint, csv, destination, batch_size, workers, limit):
    dataset = FingerprintDataset(str(csv), augment_images=False)
    count = min(len(dataset), limit) if limit else len(dataset)
    if not count:
        raise ValueError(f"Empty dataset: {csv}")
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "manifest.json"
    output = destination / "embeddings.npy"
    partial = destination / "embeddings.partial.npy"
    identity = dict(signature(checkpoint, csv), rows=count, embedding_dim=dim)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and any(state.get(k) != v for k, v in identity.items()):
        raise ValueError(f"Inputs changed for {destination}; use a new --output-dir")
    if state.get("complete"):
        vectors = np.load(output, mmap_mode="r")
        if vectors.shape != (count, dim) or vectors.dtype != np.float32:
            raise ValueError(f"Invalid completed output: {output}")
        print(f"Already complete: {output}", flush=True)
        return
    start = int(state.get("completed_rows", 0))
    if start and not partial.exists():
        # Recover interruption between final rename and completion manifest.
        if start == count and output.exists():
            save_state(state_path, dict(state, complete=True))
            return
        raise ValueError(f"Missing partial data for {destination}")
    vectors = np.lib.format.open_memmap(partial, mode="r+" if partial.exists() else "w+",
                                      dtype=np.float32, shape=(count, dim))
    metadata = dataset.df.iloc[:count].copy()
    metadata.insert(0, "embedding_row", np.arange(count))
    metadata.to_csv(destination / "metadata.csv", index=False)
    state = dict(identity, completed_rows=start, complete=False, normalized=True,
                 augmentation=False, split=csv.stem, dtype="float32")
    save_state(state_path, state)
    loader = DataLoader(Subset(dataset, range(start, count)), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    elapsed = time.monotonic()
    last_print = elapsed
    for images, _ in loader:
        batch = model(images.to(device)).cpu().numpy().astype(np.float32)
        if not np.isfinite(batch).all() or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-4):
            raise RuntimeError("Model produced invalid or non-unit embeddings")
        end = start + len(batch)
        vectors[start:end] = batch
        vectors.flush()
        state["completed_rows"] = end
        save_state(state_path, state)
        start = end
        now = time.monotonic()
        if now - last_print >= 15 or end == count:
            print(f"{checkpoint.parent.name}/{csv.stem}: {end}/{count} ({now-elapsed:.0f}s)", flush=True)
            last_print = now
    del vectors
    partial.replace(output)
    save_state(state_path, dict(state, complete=True))
    print(f"Saved {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint", type=Path, action="append", help="Explicit checkpoint; repeatable")
    parser.add_argument("--all-checkpoints", action="store_true", help="Export every checkpoint matching --checkpoint-kind")
    parser.add_argument("--checkpoint-kind", choices=("best", "latest", "both"), default="best")
    parser.add_argument("--csv", type=Path, action="append", help="Split CSV; repeatable (default train/val/test)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test rows per split; use a separate output directory")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.limit < 0 or args.threads < 1:
        parser.error("Invalid batch size, worker count, limit, or thread count")
    torch.set_num_threads(args.threads)
    if args.checkpoint and args.all_checkpoints:
        parser.error("Use either --checkpoint or --all-checkpoints")
    checkpoints = args.checkpoint or (sorted(p for p in args.checkpoint_root.rglob("*.pt")
                                           if any(p.stem == k or p.stem.startswith(k + " (")
                                                  for k in (("best", "latest") if args.checkpoint_kind == "both"
                                                            else (args.checkpoint_kind,)))) if args.all_checkpoints
                                     else [Path("checkpoints/densenet121_arcface/best.pt")])
    csvs = args.csv or [Path(f"data/processed/{s}/{s}.csv") for s in ("train", "val", "test")]
    if not checkpoints or any(not p.is_file() for p in [*checkpoints, *csvs]):
        parser.error("Missing checkpoints or processed split CSVs")
    print(f"Exporting {len(checkpoints)} checkpoints across {len(csvs)} splits", flush=True)
    for checkpoint in checkpoints:
        model, device, backbone, dim = load_model(str(checkpoint), args.device)
        # Include relative directories, filename, and path digest to avoid name collisions.
        relative = checkpoint.relative_to(args.checkpoint_root) if checkpoint.is_relative_to(args.checkpoint_root) else checkpoint
        name = "__".join(relative.with_suffix("").parts[-3:])
        name += "-" + hashlib.sha256(str(checkpoint.resolve()).encode()).hexdigest()[:8]
        print(f"Loading {checkpoint}: {backbone}, {dim} dimensions, {device}", flush=True)
        for csv in csvs:
            export_split(model, device, dim, checkpoint, csv, args.output_dir / name / csv.stem,
                         args.batch_size, args.workers, args.limit)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
