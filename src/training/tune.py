"""
Hyperparameter search.

config.yaml  = constants shared by every run (epochs, batch size, dim, paths).
tune_grid.yaml = only the keys that change per trial; those overwrite config.

    python -m src.training.tune --config src/training/config.yaml --grid src/training/tune_grid.yaml --dry-run
    python -m src.training.tune --config src/training/config.yaml --grid src/training/tune_grid.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
from datetime import datetime

import yaml

import torch

from src.training.eval import _log
from src.training.train import load_config, train


FIXED_EMBEDDING_DIM = 512


def _set_dot(cfg: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def _get_dot(cfg: dict, path: str):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_grid(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: dict, overrides: dict | None) -> dict:
    """Copy cfg and apply dotted-key overlays. embedding_dim stays 512."""
    out = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        if key in ("name",):
            continue
        if key in ("model.embedding_dim", "embedding_dim") or (
            key == "model" and isinstance(value, dict) and "embedding_dim" in value
        ):
            if value != FIXED_EMBEDDING_DIM:
                raise ValueError(
                    f"embedding_dim is fixed at {FIXED_EMBEDDING_DIM}, got {key}={value}"
                )
            continue
        _set_dot(out, key, value)
    out.setdefault("model", {})["embedding_dim"] = FIXED_EMBEDDING_DIM
    return out


def trial_overrides(trial: dict) -> dict:
    return {k: v for k, v in trial.items() if k != "name"}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--config", default="src/training/config.yaml")
    parser.add_argument("--grid", default="src/training/tune_grid.yaml")
    parser.add_argument("--backbone", default=None,
                        choices=["resnet18", "densenet121", "vgg16"],
                        help="Which backbone to tune (default: config.yaml model.backbone)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print merged configs only; do not train")
    args = parser.parse_args()

    base = load_config(args.config)
    grid = load_grid(args.grid)
    trials = grid.get("trials") or []
    if not trials:
        raise SystemExit(f"No trials in {args.grid}")

    shared = grid.get("shared_overrides") or {}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(
        base["output"].get("log_dir", "outputs/logs"), f"tune_{stamp}.csv"
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    backbone = args.backbone or base["model"]["backbone"]
    _log(f"\n  Base config : {args.config}")
    _log(f"  Tune grid   : {args.grid}  ({len(trials)} trials)")
    _log(f"  Backbone    : {backbone}   (dim={FIXED_EMBEDDING_DIM})")
    _log("  Config constants: "
         f"epochs={base['training']['epochs']}, "
         f"frozen={base['training']['frozen_epochs']}, "
         f"batch_size={base['training']['batch_size']}")
    if shared:
        _log(f"  Shared overlays (all trials): {shared}")
    _log(f"  Leaderboard -> {out_csv}\n")

    extra_keys = sorted({k for t in trials for k in t if k != "name"})
    fieldnames = ["name", *extra_keys, "best_val_eer", "ckpt_dir", "status"]

    if not args.dry_run:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows = []
    for i, trial in enumerate(trials, start=1):
        trial_name = trial.get("name") or f"trial_{i}"
        name = f"{backbone}_{trial_name}"
        cfg = apply_overrides(base, shared)
        cfg = apply_overrides(cfg, trial_overrides(trial))
        cfg["model"]["backbone"] = backbone
        cfg["output"]["checkpoint_dir"] = os.path.join(
            cfg["output"]["checkpoint_dir"], "tune"
        )
        cfg["output"]["log_dir"] = os.path.join(
            cfg["output"].get("log_dir", "outputs/logs"), "tune"
        )

        _log(f"\n{'=' * 60}")
        _log(f"  Trial {i}/{len(trials)}  {name}")
        _log(f"{'=' * 60}")
        overlay = trial_overrides(trial)
        if overlay:
            for key, value in overlay.items():
                _log(f"    overwrite  {key} = {value}  "
                     f"(config was {_get_dot(base, key)})")
        else:
            _log("    overwrite  (none - using config.yaml as-is)")
        _log(f"    resolved   lr={cfg['training']['learning_rate']}  "
             f"bb_scale={cfg['training']['backbone_lr_scale']}  "
             f"wd={cfg['training']['weight_decay']}  "
             f"epochs={cfg['training']['epochs']}  "
             f"frozen={cfg['training']['frozen_epochs']}  "
             f"dim={cfg['model']['embedding_dim']}")
        _log(f"    arcface    margin={cfg['loss'].get('kwargs', {}).get('margin')}  "
             f"scale={cfg['loss'].get('kwargs', {}).get('scale')}")

        dest_dir = os.path.join(cfg["output"]["checkpoint_dir"], name)
        latest = os.path.join(dest_dir, "latest.pt")
        _log(f"    writes to  {dest_dir}")

        row = {k: overlay.get(k, "") for k in extra_keys}
        row["name"] = name
        if os.path.isfile(latest):
            prev = torch.load(latest, map_location="cpu", weights_only=False)
            prev_epoch = int(prev.get("epoch", 0))
            if prev_epoch >= cfg["training"]["epochs"]:
                _log(f"    SKIP already finished epoch {prev_epoch} >= "
                     f"{cfg['training']['epochs']}")
                row["best_val_eer"] = f"{prev.get('val_eer', float('nan')):.6f}"
                row["ckpt_dir"] = dest_dir
                row["status"] = "skipped-complete"
                rows.append(row)
                continue

        if args.dry_run:
            row["best_val_eer"] = ""
            row["ckpt_dir"] = os.path.join(cfg["output"]["checkpoint_dir"], name)
            row["status"] = "dry-run"
            rows.append(row)
            continue

        try:
            result = train(
                cfg,
                backbone_override=backbone,
                run_name_override=name,
            )
            row["best_val_eer"] = f"{result['best_val_eer']:.6f}"
            row["ckpt_dir"] = result["ckpt_dir"]
            row["status"] = "ok"
        except Exception as exc:
            _log(f"  FAILED: {exc}")
            row["best_val_eer"] = ""
            row["ckpt_dir"] = ""
            row["status"] = f"error: {exc}"

        rows.append(row)
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    _log("\n  Leaderboard")
    ok = [r for r in rows if r.get("best_val_eer")]
    ok.sort(key=lambda r: float(r["best_val_eer"]))
    for r in ok:
        _log(f"    {r['best_val_eer']}  {r['name']}")
    if args.dry_run:
        _log("  (dry-run: nothing trained)")
    else:
        _log(f"  Saved -> {out_csv}")
    _log("")


if __name__ == "__main__":
    main()
