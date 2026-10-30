"""Evaluate individual candidate ranks using held-out Pinecone image archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pinecone_store import connect


def run(root, output, seed, people, workers):
    rng = np.random.default_rng(seed)
    data, vectors, namespaces = {}, {}, {}
    candidates = []
    index_name = None
    for split in ("val", "test"):
        folder = root / split
        meta = pd.read_csv(folder / "metadata.csv", dtype={"class_id": str}).fillna("")
        info = json.loads((folder / "pinecone.json").read_text(encoding="utf-8"))
        if not info.get("complete") or (index_name and info["index"] != index_name):
            raise ValueError("Both archives must be verified in the same Pinecone index")
        index_name = info["index"]
        data[split] = meta.set_index("embedding_row", drop=False)
        vectors[split] = np.load(folder / "embeddings.npy", mmap_mode="r")
        namespaces[split] = info["namespace"]
        for subject, group in meta.groupby("subject_id"):
            if group.class_id.nunique() == 10:
                candidates.append((split, str(subject)))
    subjects = [subject for _, subject in candidates]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Subject overlap between validation and test splits")
    if len(candidates) < people:
        raise ValueError("Not enough eligible subjects")
    chosen = rng.choice(len(candidates), size=people, replace=False)
    tasks = []
    for pos in chosen:
        split, subject = candidates[int(pos)]
        group = data[split][data[split].subject_id.astype(str) == subject]
        for class_id, finger in group.groupby("class_id", sort=True):
            row = finger.iloc[int(rng.integers(len(finger)))]
            tasks.append((split, subject, class_id, int(row.embedding_row)))
    index = connect(index_name, int(vectors['test'].shape[1]))

    def query(task):
        split, subject, label, row = task
        response = index.query(namespace=namespaces[split], vector=vectors[split][row].tolist(),
                               top_k=11, include_values=False, include_metadata=False)
        matches = list(response.matches)
        self_positions = [i for i, m in enumerate(matches) if m.id == str(row)]
        remaining = [m for m in matches if m.id != str(row)]
        if len(remaining) < 5:
            raise ValueError("Pinecone returned too few candidates")
        out = {"split": split, "subject_id": subject, "class_id": label,
               "embedding_row": row, "self_original_rank": self_positions[0] + 1 if self_positions else None}
        for rank, position in ((2, 0), (4, 2), (6, 4)):
            match = remaining[position]
            matched_id = int(match.id)
            if matched_id not in data[split].index:
                raise ValueError("Pinecone ID not found in corresponding local metadata")
            matched_label = str(data[split].loc[matched_id, "class_id"])
            out.update({f"rank{rank}_class_id": matched_label, f"rank{rank}_score": float(match.score),
                        f"rank{rank}_correct": matched_label == label,
                        f"rank{rank}_original_rank": next(i + 1 for i, m in enumerate(matches) if m.id == match.id)})
        return out

    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(query, tasks):
            results.append(result)
            if len(results) % 100 == 0:
                print(f"Completed {len(results)}/{len(tasks)} queries", flush=True)
    detail = pd.DataFrame(results)
    per_person = detail.groupby(["split", "subject_id"], sort=True).agg(
        queries=("class_id", "size"), rank2_matches=("rank2_correct", "sum"),
        rank4_matches=("rank4_correct", "sum"), rank6_matches=("rank6_correct", "sum")).reset_index()
    for rank in (2, 4, 6):
        per_person[f"rank{rank}_accuracy"] = per_person[f"rank{rank}_matches"] / per_person.queries
    detail.to_csv(output / "query_results.csv", index=False)
    per_person.to_csv(output / "person_scores.csv", index=False)
    summary = {"seed": seed, "people": people, "queries": len(detail), "index": index_name,
               "method": "Remove exact query ID; compare 1st, 3rd, 5th remaining candidate by finger class_id",
               "caveat": "Validation/test subjects; searches within same split and include other images of each finger. Not open-set evaluation.",
               "self_not_first": int((detail.self_original_rank != 1).sum()),
               "average_accuracy": {f"rank{r}": float(per_person[f"rank{r}_accuracy"].mean()) for r in (2, 4, 6)}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/embeddings/final_densenet/densenet121_arcface__best-2228dbf0"))
    parser.add_argument("--output", type=Path, default=Path("outputs/person_evaluation/densenet121_arcface"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--people", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.people < 1 or not 1 <= args.workers <= 16:
        parser.error("people must be positive and workers must be 1..16")
    run(args.root, args.output, args.seed, args.people, args.workers)


if __name__ == "__main__":
    main()
