"""Query a saved gallery by held-out row index; optional grounded Ollama response."""
import argparse
import json
import urllib.request

from src.data.dataset import FingerprintDataset
from .core import load_model, embed_rows, load_index


def search(index, gallery, vector, k=5):
    scores, indices = index.search(vector, min(k, index.ntotal))
    return [{"rank": rank + 1, "class_id": str(gallery.iloc[int(j)]["class_id"]),
             "csv_row": int(gallery.iloc[int(j)]["csv_row"]), "cosine": float(s)}
            for rank, (s, j) in enumerate(zip(scores[0], indices[0])) if j >= 0]


def grounded_summary(matches, threshold):
    if not matches:
        return "No enrolled candidates."
    top = matches[0]
    decision = "candidate above threshold" if top["cosine"] >= threshold else "no match above threshold"
    return (f"Top retrieved finger: {top['class_id']} (cosine={top['cosine']:.4f}); "
            f"decision: {decision}. This is an automated retrieval result, not verified identity.")


def ollama_answer(matches, threshold, model_name):
    """Optional generation only; never let the LLM change the match decision."""
    facts = {"threshold": threshold, "matches": matches,
             "deterministic_result": grounded_summary(matches, threshold)}
    prompt = ("You are describing fingerprint search results. Use ONLY the JSON facts. "
              "Do not invent names, confidence probabilities, biographical details, or extra matches. "
              "Explain that cosine is not probability and that this is a candidate match, not conclusive identity. "
              "If below threshold, explicitly say no match. Keep it under 100 words.\n"
              + json.dumps(facts))
    data = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)["response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="outputs/retrieval/densenet121")
    ap.add_argument("--query-number", type=int, default=0,
                    help="Zero-based position in saved queries.csv, not raw CSV row number")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Validation-calibrated cosine threshold; omit until calibrated")
    ap.add_argument("--ollama-model", default=None, help="Optional installed Ollama model name")
    args = ap.parse_args()
    index, gallery, queries, manifest = load_index(args.index_dir)
    if not (0 <= args.query_number < len(queries)):
        raise ValueError(f"--query-number must be 0..{len(queries)-1}")
    data = FingerprintDataset(manifest["test_csv"], augment_images=False)
    model, device, backbone, dim = load_model(manifest["checkpoint"])
    if index.d != dim:
        raise ValueError(f"Index dim {index.d} != model dim {dim}")
    row = queries.iloc[args.query_number]
    true_id = str(row["class_id"])
    vec = embed_rows(model, device, data, [int(row["csv_row"])], batch_size=1)
    matches = search(index, gallery, vec, args.top_k)
    print(f"Query #{args.query_number}, CSV row {row['csv_row']}, known label {true_id}")
    for m in matches:
        print(f"  {m['rank']:>2} | {m['class_id']:<28} | cosine={m['cosine']:.4f}")
    print("Rank-1 correct:", bool(matches and matches[0]["class_id"] == true_id))
    if args.threshold is None:
        print("Threshold decision: NOT CALIBRATED (ranking only)")
    else:
        print("Threshold decision:", grounded_summary(matches, args.threshold))
    if args.ollama_model:
        if args.threshold is None:
            raise ValueError("Use --threshold calibrated on VAL before generating a match decision")
        try:
            print("\nLocal RAG explanation:\n", ollama_answer(matches, args.threshold, args.ollama_model))
        except Exception as e:
            print(f"Ollama unavailable: {e}; deterministic result above remains available")

if __name__ == "__main__":
    main()
