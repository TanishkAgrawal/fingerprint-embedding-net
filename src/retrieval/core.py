"""Common dataset, checkpoint, and index utilities.

Run modules from the repository root via python -m src.retrieval.<module>.
"""
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from src.data.dataset import FingerprintDataset
from src.models.backbone import FingerprintEmbeddingNet


def load_model(checkpoint: str, device: str = "auto"):
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       "cpu" if device == "auto" else device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    backbone = ckpt.get("backbone", "densenet121")
    # Infer dimension from the projection head only if checkpoint metadata lacks it.
    dim = ckpt.get("embedding_dim", 512)
    model = FingerprintEmbeddingNet(backbone_name=backbone,
                                   embedding_dim=int(dim), pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(dev).eval()
    return model, dev, backbone, int(dim)


def read_split(csv_path: str):
    df = pd.read_csv(csv_path).fillna("")
    if "class_id" not in df.columns:
        raise ValueError(f"CSV must have 'class_id'; columns: {df.columns.tolist()}")
    df["class_id"] = df["class_id"].astype(str)
    if df["class_id"].eq("").any():
        raise ValueError("Some CSV rows lack class_id")
    return df


def real_mask(df: pd.DataFrame):
    """Identify real images using explicit metadata or SOCOFing source path.

    Returns None if real/altered classification is not identifiable.
    """
    for col in ("alteration", "alteration_type", "altered_type", "category", "dataset_type", "split_type", "image_type", "type"):
        if col in df:
            v = df[col].astype(str).str.strip().str.lower()
            mask = v.isin(["real", "original", "unaltered", "none"])
            if mask.any():
                return mask.to_numpy()
    for col in ("original_path", "orig_path", "raw_path", "source_path", "source_file", "file_path", "image_path", "filepath", "path"):
        if col in df:
            v = df[col].astype(str).str.replace("\\", "/", regex=False).str.lower()
            mask = v.str.contains(r"(^|/)real/", regex=True)
            if mask.any():
                return mask.to_numpy()
    return None


def split_gallery_queries(df):
    """One enrollment per finger. Other images become distinct query rows."""
    real = real_mask(df)
    picks = []
    for _, group in df.groupby("class_id", sort=False):
        options = group.index.to_numpy()
        if real is not None:
            originals = [int(i) for i in options if real[int(i)]]
            if originals:
                picks.append(originals[0]); continue
        picks.append(int(options[0]))
    enrolled = set(picks)
    queries = [i for i in range(len(df)) if i not in enrolled]
    if not queries:
        raise ValueError("No separate query images: each class has only one sample")
    if real is None:
        print("WARNING: original/altered status not detectable; using first row per finger as gallery.")
        print("         Review CSV columns and gallery selection before reporting Real-to-Altered results.")
    return picks, queries, real is not None


@torch.inference_mode()
def embed_rows(model, device, dataset, indices, batch_size=32):
    vectors = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        imgs = []
        for i in chunk:
            item = dataset[int(i)]
            img = item[0] if isinstance(item, (tuple, list)) else item["image"]
            imgs.append(img)
        batch = torch.stack(imgs).to(device)
        vectors.append(model(batch).detach().cpu().numpy().astype("float32"))
        if start == 0 or start + len(chunk) >= len(indices):
            print(f"Embedded {start + len(chunk)}/{len(indices)}")
    result = np.concatenate(vectors, axis=0)
    faiss.normalize_L2(result)
    return result


def save_index(index_dir, vectors, gallery_rows, query_rows, metadata, manifest):
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    faiss.write_index(index, str(root / "index.faiss"))
    metadata.iloc[gallery_rows].assign(csv_row=gallery_rows).to_csv(root / "gallery.csv", index=False)
    metadata.iloc[query_rows].assign(csv_row=query_rows).to_csv(root / "queries.csv", index=False)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_index(index_dir):
    root = Path(index_dir)
    index = faiss.read_index(str(root / "index.faiss"))
    gallery = pd.read_csv(root / "gallery.csv").fillna("")
    queries = pd.read_csv(root / "queries.csv").fillna("")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return index, gallery, queries, manifest
