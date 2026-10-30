"""Read-only evaluation of an existing SOCOFing fingerprint Pinecone gallery.

Usage: python -m src.retrieval.evaluate_detailed --index-dir outputs/retrieval/densenet121
No training, checkpoint writes, index rebuilding or LLM inference.
"""
import argparse
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from src.data.dataset import FingerprintDataset
from .core import load_index, load_model, embed_rows, real_mask

K_VALUES = (1, 5, 10)
SEEDS = (42, 43, 44)
DIFFICULTIES = ("easy", "medium", "hard")


def _float(x):
    v = float(x)
    return v if math.isfinite(v) else None


def _summarize(values):
    vals = np.asarray(values, dtype=float)
    return {"mean": _float(vals.mean()), "std": _float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals))}


def _table(headers, rows):
    values = [[str(v) for v in row] for row in rows]
    widths = [max(len(str(h)), *(len(row[i]) for row in values)) if values else len(str(h))
              for i, h in enumerate(headers)]
    print("| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |")
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in values:
        print("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")


def _fmt(x):
    return "N/A" if x is None else f"{x:.4f}"


def _row_metrics(correct, top1_accepted=None):
    """Return per-query correctness and correctly accepted flags, if available."""
    correct = np.asarray(correct, dtype=bool)
    data = {"rank1": correct[:, 0]}
    for k in K_VALUES:
        data[f"recall_at_{k}"] = correct[:, :k].any(axis=1)
        data[f"precision_at_{k}"] = data[f"recall_at_{k}"].astype(float) / k
    if top1_accepted is not None:
        data["accepted"] = np.asarray(top1_accepted, dtype=bool)
    return data


def _aggregate(rows, indices):
    result = {"n_queries": len(indices)}
    for field in ["rank1", *[f"recall_at_{k}" for k in K_VALUES],
                  *[f"precision_at_{k}" for k in K_VALUES]]:
        result[field] = _float(np.mean(rows[field][indices]))
    if "accepted" in rows:
        accepted = rows["accepted"][indices]
        correct = rows["rank1"][indices]
        tp = int(np.count_nonzero(accepted & correct))
        fp = int(np.count_nonzero(accepted & ~correct))
        fn = int(np.count_nonzero(~accepted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        result["threshold"] = {"tp": tp, "fp": fp, "fn": fn, "tn": None,
                               "precision": precision, "recall": recall, "f1": f1,
                               "accepted_wrong": fp, "accepted_count": int(accepted.sum()),
                               "rejected_count": fn,
                               "definition": "TP=accepted/correct; FP=accepted/wrong; FN=rejected (closed-set); no TN"}
    return result


def _diff_of_record(row):
    """Determine Easy/Medium/Hard from explicit metadata or source path.

    Never infer difficulty from the raw cosine score or result correctness.
    """
    preferred = ("difficulty", "alteration_level", "alteration_difficulty", "alteration", "altered_type",
                 "category", "dataset_type", "split_type", "image_type", "type")
    paths = ("original_path", "orig_path", "raw_path", "source_path", "source_file",
             "file_path", "image_path", "filepath", "path")
    for field in (*preferred, *paths):
        if field not in row.index:
            continue
        s = str(row[field]).lower().replace("\\", "/")
        for d in DIFFICULTIES:
            if re.search(r"(?<![a-z])" + d + r"(?![a-z])", s):
                return d
    return "unknown"


def _identity_strata(queries):
    queries = queries.copy()
    queries["eval_difficulty"] = queries.apply(_diff_of_record, axis=1)
    group_to_indices = {}
    identity_strata = {}
    for identity, group in queries.groupby("class_id", sort=True):
        counts = Counter(group["eval_difficulty"])
        # Define each identity's stratum by its most frequent query difficulty.
        # Tie-break deterministic: easy, medium, hard, unknown.
        identity_strata[str(identity)] = sorted(counts, key=lambda x: (-counts[x],
            ("easy", "medium", "hard", "unknown").index(x)))[0]
        group_to_indices[str(identity)] = group.index.to_numpy(dtype=int)
    return queries, identity_strata, group_to_indices


def _choose_random(groups, seed):
    pool = np.asarray(sorted(groups))
    return sorted(np.random.default_rng(seed).choice(pool, size=10, replace=False).tolist())


def _choose_stratified(groups, strata, seed):
    rng = np.random.default_rng(seed)
    buckets = {d: sorted(k for k in groups if strata[k] == d) for d in (*DIFFICULTIES, "unknown")}
    targets = {"easy": 4, "medium": 3, "hard": 3}
    selected = []
    shortfalls = {}
    for d in DIFFICULTIES:
        pool = buckets[d]
        n = min(targets[d], len(pool))
        if n < targets[d]:
            shortfalls[d] = {"target": targets[d], "available": len(pool), "taken": n}
        if n:
            selected.extend(rng.choice(pool, size=n, replace=False).tolist())
    needed = 10 - len(selected)
    if needed:
        available = sorted(set(groups).difference(selected))
        selected.extend(rng.choice(available, size=needed, replace=False).tolist())
    return sorted(selected), shortfalls


def _sampling_metrics(ids, grouped, all_rows):
    per_finger = []
    macro_fields = ["rank1", "recall_at_1", "recall_at_5", "recall_at_10",
                    "precision_at_1", "precision_at_5", "precision_at_10"]
    if "accepted" in all_rows:
        macro_fields += ["threshold.precision", "threshold.recall", "threshold.f1"]
    for identity in ids:
        m = _aggregate(all_rows, grouped[identity])
        per_finger.append({"class_id": identity, **m})
    all_idx = np.concatenate([grouped[identity] for identity in ids])
    micro = _aggregate(all_rows, all_idx)
    macro = {"n_fingers": len(ids), "n_queries": len(all_idx)}
    for field in macro_fields:
        if field.startswith("threshold."):
            f = field.split(".", 1)[1]
            macro[field] = _float(np.mean([v["threshold"][f] for v in per_finger]))
        else:
            macro[field] = _float(np.mean([v[field] for v in per_finger]))
    return per_finger, macro, micro


def _percent(x):
    return "N/A" if x is None else f"{100*x:.2f}%"


def _print_full(full):
    print("\n### A. Full query set (full gallery)")
    records = [("Gallery", full["gallery_size"]), ("Queries", full["n_queries"]),
               ("Rank-1", _percent(full["rank1"]))]
    for k in K_VALUES:
        records.extend([(f"Recall@{k}", _percent(full[f"recall_at_{k}"])),
                        (f"Precision@{k}", _percent(full[f"precision_at_{k}"]))])
    _table(["Metric", "Value"], records)
    if "threshold" in full:
        t = full["threshold"]
        print("\nThresholded Top-1 (closed-set definitions; TN does not exist):")
        _table(["TP", "FP", "FN", "TN", "Precision", "Recall", "F1"],
               [[t["tp"], t["fp"], t["fn"], "N/A", _percent(t["precision"]),
                 _percent(t["recall"]), _percent(t["f1"])]])
    else:
        print("Threshold metrics skipped: threshold.json not found.")


def _print_sample(mode, seed, sample):
    print(f"\n### B. {mode.title()}, seed={seed}; gallery is unchanged")
    records = []
    for item in sample["per_finger"]:
        t = item.get("threshold")
        records.append([item["class_id"], sample["identity_strata"][item["class_id"]],
                        item["n_queries"], _percent(item["rank1"]),
                        _percent(item["recall_at_5"]), _percent(item["precision_at_5"]),
                        _percent(t["precision"]) if t else "N/A",
                        _percent(t["recall"]) if t else "N/A"])
    _table(["class_id", "stratum", "n", "Rank-1", "R@5", "P@5", "Threshold P", "Threshold R"], records)
    aggregate_rows = []
    for label, m in [("Macro (10 fingers)", sample["macro"]),
                     ("Micro (all queries)", sample["micro"])]:
        t = m.get("threshold", {})
        aggregate_rows.append([label, m["n_queries"], _percent(m["rank1"]),
                               _percent(m["recall_at_5"]), _percent(m["precision_at_5"]),
                               _percent(m.get("threshold.precision", t.get("precision"))),
                               _percent(m.get("threshold.recall", t.get("recall")))])
    _table(["Aggregation", "n_queries", "Rank-1", "R@5", "P@5", "Threshold P", "Threshold R"], aggregate_rows)
    if sample.get("shortfalls"):
        print("Stratum shortage; filled from available other strata:", sample["shortfalls"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="outputs/retrieval/densenet121")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-workers", type=int, default=8, help="Concurrent Pinecone requests")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not create eval_report.json")
    args = parser.parse_args()
    if args.query_workers < 1:
        parser.error("--query-workers must be positive")
    root = Path(args.index_dir)
    index, gallery, queries, manifest = load_index(root)
    if len(gallery) != index.ntotal:
        raise ValueError("Index vector count does not equal gallery metadata row count")
    if len(gallery) < 10:
        raise ValueError("Need at least 10 distinct enrolled identities for the requested sampling")
    if gallery["class_id"].astype(str).duplicated().any():
        raise ValueError("Gallery does not have exactly one image per class_id")
    if len(queries) == 0 or "csv_row" not in gallery or "csv_row" not in queries:
        raise ValueError("Missing query records or csv_row alignment metadata")
    if set(gallery["csv_row"].astype(int)) & set(queries["csv_row"].astype(int)):
        raise ValueError("Enrolled CSV rows appear in the query set; refusing leaked evaluation")
    gallery_ids = gallery["class_id"].astype(str)
    queries["class_id"] = queries["class_id"].astype(str)
    if not set(queries["class_id"]).issubset(set(gallery_ids)):
        raise ValueError("Some queries have no enrolled class_id; protocol is not closed-set")
    if len(gallery_ids) != gallery_ids.nunique():
        raise ValueError("Gallery not unique by class_id")
    for path_col in ("image_path", "file_path", "filepath", "processed_path", "npy_path"):
        if path_col in gallery and path_col in queries:
            gpaths = set(gallery[path_col].astype(str).str.lower().str.replace("\\", "/", regex=False)) - {"", "nan"}
            qpaths = set(queries[path_col].astype(str).str.lower().str.replace("\\", "/", regex=False)) - {"", "nan"}
            if gpaths & qpaths:
                raise ValueError(f"Gallery and query image paths overlap in column {path_col}")
    q_real = real_mask(queries)
    if q_real is not None and np.asarray(q_real).any():
        raise ValueError("Query set contains an image explicitly identified as Real; expected altered query images only")
    mask = real_mask(gallery)
    real_verified = bool(mask is not None and np.asarray(mask).all())
    if not real_verified:
        print("WARNING: Cannot prove all gallery rows are Real from metadata. Confirm enrollment source before reporting Real-to-Altered results.")
    queries, strata, grouped = _identity_strata(queries.reset_index(drop=True))
    if len(grouped) < 10:
        raise ValueError("Fewer than 10 distinct query class_ids")
    print(f"Gallery={index.ntotal}; query images={len(queries)}; query identities={len(grouped)}")
    print("Query difficulty records:", dict(Counter(queries["eval_difficulty"])))
    print("Primary identity strata:", dict(Counter(strata.values())))
    print("Definition: identity stratum = most frequent recorded query difficulty; ties easy > medium > hard > unknown")
    print("Precision@k = (1/k) if the true class appears in top-k, otherwise 0 (one gallery vector per class)")

    model, device, backbone, dim = load_model(manifest["checkpoint"])
    if dim != index.d or int(manifest.get("embedding_dim", dim)) != dim:
        raise ValueError("Model checkpoint embedding dimension and Pinecone index do not match")
    if manifest.get("backbone") and manifest["backbone"] != backbone:
        raise ValueError("Manifest backbone mismatches checkpoint")
    dataset = FingerprintDataset(manifest["test_csv"], augment_images=False)
    if len(dataset) <= max(queries["csv_row"].astype(int)):
        raise ValueError("Query indices exceed test dataset length")
    vec = embed_rows(model, device, dataset, queries["csv_row"].astype(int).tolist(), args.batch_size)
    k = min(max(K_VALUES), int(index.ntotal))
    def search_one(vector):
        return index.search(vector[None, :], k)
    results = []
    with ThreadPoolExecutor(max_workers=args.query_workers) as pool:
        # map preserves query order even when requests finish out of order.
        for result in pool.map(search_one, vec):
            results.append(result)
            if len(results) % 500 == 0 or len(results) == len(vec):
                print(f"Searched {len(results)}/{len(vec)} queries", flush=True)
    scores = np.concatenate([result[0] for result in results])
    idx = np.concatenate([result[1] for result in results])
    found = gallery_ids.to_numpy()[idx] == queries["class_id"].to_numpy()[:, None]

    threshold_path = root / "threshold.json"
    threshold_info = None
    accepted = None
    if threshold_path.exists():
        threshold_info = json.loads(threshold_path.read_text(encoding="utf-8"))
        if Path(threshold_info["checkpoint"]).resolve() != Path(manifest["checkpoint"]).resolve():
            raise ValueError("Threshold calibration checkpoint mismatches gallery checkpoint")
        threshold = float(threshold_info["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("Invalid non-finite threshold")
        if threshold_info.get("backbone") and threshold_info["backbone"] != backbone:
            raise ValueError("Threshold calibration backbone mismatches loaded checkpoint")
        if "embedding_dim" in threshold_info and int(threshold_info["embedding_dim"]) != dim:
            raise ValueError("Threshold calibration embedding dimension mismatches index")
        accepted = scores[:, 0] >= threshold
        print(f"Threshold={threshold:.6f}, sourced ONLY from {threshold_path} (pairwise validation-calibrated)")
    rows = _row_metrics(found, accepted)
    full = _aggregate(rows, np.arange(len(queries), dtype=int))
    full["gallery_size"] = int(len(gallery))
    _print_full(full)

    sampling = {}
    summary = {}
    for mode in ("random", "stratified"):
        trials = []
        for seed in SEEDS:
            if mode == "random":
                ids = _choose_random(grouped, seed)
                shortfalls = {}
            else:
                ids, shortfalls = _choose_stratified(grouped, strata, seed)
            per_finger, macro, micro = _sampling_metrics(ids, grouped, rows)
            trial = {"seed": seed, "selected_class_ids": ids,
                     "identity_strata": {x: strata[x] for x in ids},
                     "strata_counts": dict(Counter(strata[x] for x in ids)),
                     "shortfalls": shortfalls, "per_finger": per_finger,
                     "macro": macro, "micro": micro,
                     "gallery_size": int(index.ntotal)}
            trials.append(trial)
            _print_sample(mode, seed, trial)
        sampling[mode] = trials
        names = ("rank1", "recall_at_1", "recall_at_5", "recall_at_10",
                 "precision_at_1", "precision_at_5", "precision_at_10")
        if accepted is not None:
            names += ("threshold.precision", "threshold.recall", "threshold.f1")
        summary[mode] = {name: _summarize([t["macro"][name] for t in trials]) for name in names}

    print("\n### Macro mean ± sample standard deviation across seeds 42, 43, 44")
    overview = []
    for mode in ("random", "stratified"):
        for name, stats in summary[mode].items():
            overview.append([mode, name, f"{stats['mean']:.4f} ± {stats['std']:.4f}",
                             f"{stats['mean']*100:.2f}% ± {stats['std']*100:.2f}%"])
    _table(["Sampling", "Macro metric", "Mean ± std", "Percent"], overview)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "index_dir": str(root), "backbone": backbone, "embedding_dim": dim,
        "checkpoint_read_only": manifest["checkpoint"], "gallery_size": int(index.ntotal),
        "full_gallery_used_for_every_search": True,
        "retrieval_backend": "pinecone",
        "search_is_approximate": True,
        "gallery_all_real_proved_by_metadata": real_verified,
        "query_difficulty_counts": dict(Counter(queries["eval_difficulty"])),
        "identity_strata_counts": dict(Counter(strata.values())),
        "difficulty_stratification_rule": "modal query-image difficulty per identity; ties easy > medium > hard > unknown",
        "sampling_target": {"easy": 4, "medium": 3, "hard": 3},
        "sampling_seeds": list(SEEDS),
        "precision_at_k_definition": "1/k if correct class_id in Top-k, otherwise 0; exactly one gallery image per class_id",
        "threshold": threshold_info,
        "threshold_warning": "Closed-set accepted/wrong and rejected counting; pairwise-calibrated threshold is NOT validated open-set FAR",
        "threshold_definition": "TP=accepted and correct; FP=accepted and wrong; FN=rejected; TN not defined",
        "evaluation_full": full, "sampling": sampling, "macro_across_seeds": summary,
        "llm_used_for_identity": False,
        "training_performed": False,
        "checkpoints_modified": False,
    }
    if not args.no_write:
        dest = root / "eval_report.json"
        dest.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"\nSaved -> {dest}")


if __name__ == "__main__":
    main()
