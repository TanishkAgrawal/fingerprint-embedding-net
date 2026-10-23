"""Build a held-out fingerprint enrollment gallery (one fingerprint per class)."""
import argparse
from src.data.dataset import FingerprintDataset
from .core import load_model, read_split, split_gallery_queries, embed_rows, save_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-csv", default="data/processed/test/test.csv")
    ap.add_argument("--index-dir", default="outputs/retrieval/densenet121")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    df = read_split(args.test_csv)
    dataset = FingerprintDataset(args.test_csv, augment_images=False)
    if len(dataset) != len(df):
        raise RuntimeError("CSV/dataset length mismatch; review FingerprintDataset ordering")
    gallery_ids, query_ids, detected = split_gallery_queries(df)
    print(f"Enrollments: {len(gallery_ids)}; queries: {len(query_ids)}; Real detected: {detected}")
    model, device, backbone, dim = load_model(args.checkpoint)
    gallery_vecs = embed_rows(model, device, dataset, gallery_ids, args.batch_size)
    save_index(args.index_dir, gallery_vecs, gallery_ids, query_ids, df,
               {"checkpoint": args.checkpoint, "test_csv": args.test_csv,
                "backbone": backbone, "embedding_dim": dim,
                "gallery_size": len(gallery_ids), "query_size": len(query_ids),
                "real_images_detected": detected,
                "note": "Each CSV row is one fingerprint IMAGE; class_id identifies a finger."})
    print(f"Saved FAISS index and metadata to {args.index_dir}")

if __name__ == "__main__":
    main()
