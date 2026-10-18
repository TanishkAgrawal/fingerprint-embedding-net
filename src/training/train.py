"""
Shared training loop for the fingerprint embedding project.

Works identically for all three backbones (resnet18, densenet121, vgg16) and
both loss functions (arcface, triplet). All hyperparameters come from a YAML
config so nothing is hardcoded here.

Training strategy (progressive unfreezing):
  Phase 1 — Frozen backbone (config.training.frozen_epochs epochs)
      Only the embedding head is trained. Fast, low memory.
  Phase 2 — Full fine-tuning (remaining epochs)
      Backbone unfreezes with backbone_lr_scale × base_lr. Loss module
      (ArcFace weight matrix) also included in the optimizer.
"""

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data.dataset import get_dataloader
from src.models.backbone import FingerprintEmbeddingNet
from src.models.losses import get_loss
from src.training.eval import verification_metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Print immediately so long epochs don't look hung in the terminal."""
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    _log(f"  [ckpt] saved -> {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer, scaler,
                    criterion: nn.Module, device):
    """Load a checkpoint and return the starting epoch."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    if "criterion" in ckpt and hasattr(criterion, "loss_fn"):
        criterion.loss_fn.load_state_dict(ckpt["criterion"])
    start_epoch = ckpt["epoch"] + 1
    _log(f"  [ckpt] resumed from epoch {ckpt['epoch']}  (loss={ckpt['val_loss']:.4f})")
    return start_epoch


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def resolve_device(cfg: dict) -> torch.device:
    """
    Pick the training device from config.

    training.device:
      - 'cuda' (default): use GPU; exit with install hints if unavailable
      - 'auto': GPU if available, otherwise CPU
      - 'cpu': force CPU
    """
    preference = cfg.get("training", {}).get("device", "cuda").lower().strip()

    if preference == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        idx = cfg.get("training", {}).get("cuda_device", 0)
        return torch.device(f"cuda:{idx}")

    if preference == "auto":
        return torch.device("cpu")

    _log("\nERROR: CUDA requested (training.device: cuda) but torch.cuda.is_available() is False.")
    _log(f"  Python  : {sys.executable}")
    _log(f"  PyTorch : {torch.__version__}")
    if "+cpu" in torch.__version__:
        _log("  Cause   : CPU-only PyTorch build (version contains '+cpu').")
    _log("\n  Reinstall GPU PyTorch in this environment, then retry:")
    _log("    pip uninstall -y torch torchvision")
    _log("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
    _log("\n  Or set training.device: auto to allow CPU (very slow).\n")
    sys.exit(1)


@contextlib.contextmanager
def _autocast(device: torch.device):
    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            yield
    else:
        yield


class _CpuScaler:
    """No-op scaler so the train loop stays identical on CPU."""

    def scale(self, loss):
        return loss

    def unscale_(self, _optimizer) -> None:
        pass

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, _state) -> None:
        pass


def build_grad_scaler(device: torch.device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return _CpuScaler()


# ── Epoch functions ───────────────────────────────────────────────────────────

def train_one_epoch(
    model, criterion, loader, optimizer, scaler, device,
    epoch: int, total_epochs: int, log_interval: int = 50,
):
    model.train()
    total_loss, n_batches = 0.0, 0
    n_total = len(loader)
    pin = device.type == "cuda"

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=pin)
        labels = labels.to(device, non_blocking=pin)

        optimizer.zero_grad(set_to_none=True)

        with _autocast(device):
            embeddings = model(imgs)
            loss       = criterion(embeddings, labels)

        scaler.scale(loss).backward()
        # Gradient clipping: prevents exploding gradients, especially useful
        # during the first few batches of phase 2 when the backbone unfreezes.
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

        step = batch_idx + 1
        if step == 1 or step % log_interval == 0 or step == n_total:
            avg = total_loss / n_batches
            gpu_bit = ""
            if device.type == "cuda":
                gpu_bit = f"  gpu_mem={torch.cuda.memory_allocated(device) / (1024 ** 2):.0f}MB"
            _log(
                f"    epoch {epoch}/{total_epochs}  train "
                f"{step}/{n_total}  avg_loss={avg:.4f}  last={loss.item():.4f}"
                f"{gpu_bit}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, criterion, loader, device, log_interval: int = 100):
    model.eval()
    total_loss, n_batches = 0.0, 0
    n_total = len(loader)
    pin = device.type == "cuda"

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=pin)
        labels = labels.to(device, non_blocking=pin)

        with _autocast(device):
            embeddings = model(imgs)
            loss       = criterion(embeddings, labels)

        total_loss += loss.item()
        n_batches  += 1

        step = batch_idx + 1
        if step == 1 or step % log_interval == 0 or step == n_total:
            avg = total_loss / n_batches
            _log(f"    val {step}/{n_total}  avg_loss={avg:.4f}")

    return total_loss / max(n_batches, 1)


# ── Optimizer builder ─────────────────────────────────────────────────────────

def build_optimizer(model, criterion, cfg: dict, phase: int):
    """
    Build an AdamW optimizer for the given training phase.

    Phase 1 (frozen backbone): only the embedding head + ArcFace matrix.
    Phase 2 (full fine-tune) : backbone (lower LR) + head + ArcFace matrix.
    """
    base_lr     = cfg["training"]["learning_rate"]
    bb_scale    = cfg["training"]["backbone_lr_scale"]
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    loss_type   = cfg["loss"]["type"]

    if phase == 1:
        # Only the embedding head (backbone is frozen → its params have
        # requires_grad=False and won't appear in a default param scan)
        param_groups = [
            {"params": list(model.embedding_head.parameters()),
             "lr": base_lr, "name": "embedding_head"},
        ]
    else:
        # Backbone at lower LR + head at full LR
        param_groups = model.get_param_groups(base_lr, bb_scale)

    # Always include ArcFace weight matrix if present
    if loss_type == "arcface":
        param_groups += criterion.get_optimizer_params(lr=base_lr)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]
    return optimizer


# ── Main training function ────────────────────────────────────────────────────

def train(cfg: dict, backbone_override: str | None = None,
          loss_override: str | None = None):

    # ── Config resolution ──────────────────────────────────────────────
    backbone_name = backbone_override or cfg["model"]["backbone"]
    loss_type     = loss_override     or cfg["loss"]["type"]
    embedding_dim = cfg["model"]["embedding_dim"]
    total_epochs  = cfg["training"]["epochs"]
    frozen_epochs = cfg["training"]["frozen_epochs"]
    batch_size    = cfg["training"]["batch_size"]
    num_workers   = cfg["data"].get("num_workers", 0)
    log_interval  = cfg["training"].get("log_interval", 50)

    # Run identifier used for checkpoints + logs
    run_name = f"{backbone_name}_{loss_type}"

    # ── Device ────────────────────────────────────────────────────────
    device = resolve_device(cfg)
    _log(f"\n{'='*60}")
    _log(f"  Run      : {run_name}")
    _log(f"  Python   : {sys.executable}")
    _log(f"  PyTorch  : {torch.__version__}")
    _log(f"  Device   : {device}")
    if device.type == "cuda":
        idx = device.index or 0
        _log(f"  GPU      : {torch.cuda.get_device_name(idx)}")
        _log("  Task Mgr : NVIDIA graph at 0% is normal if it shows the 3D engine.")
        _log("             Click GPU 1 (NVIDIA) → change the graph to CUDA / Compute.")
        _log("             AMD Radeon is the iGPU and is not used by PyTorch.")
    elif cfg.get("training", {}).get("device", "cuda").lower() == "auto":
        _log("  NOTE     : CUDA not available — running on CPU (training.device: auto).")
    _log(f"  Epochs   : {total_epochs}  (frozen={frozen_epochs}, "
         f"finetune={total_epochs - frozen_epochs})")
    _log(f"  Log every: {log_interval} train batches")
    _log(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────
    train_csv = cfg["data"]["train_csv"]
    val_csv   = cfg["data"]["val_csv"]
    alt_filter = cfg["data"].get("alteration_filter", None)

    train_loader, num_classes = get_dataloader(
        train_csv, split="train", batch_size=batch_size,
        num_workers=num_workers, alteration_filter=alt_filter,
    )
    val_loader, _ = get_dataloader(
        val_csv, split="val", batch_size=batch_size * 2,
        num_workers=num_workers,
    )
    _log(f"  Train batches : {len(train_loader)}  |  classes : {num_classes}")
    _log(f"  Val   batches : {len(val_loader)}\n")

    # ── Model + Loss ──────────────────────────────────────────────────
    _log("  Building model and loss...")
    model     = FingerprintEmbeddingNet(backbone_name, embedding_dim).to(device)
    criterion = get_loss(loss_type, num_classes=num_classes,
                         embedding_dim=embedding_dim,
                         **cfg["loss"].get("kwargs", {})).to(device)
    model.freeze_backbone()
    _log(f"  {model}")
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        _log(f"  GPU mem after model load: {allocated:.0f} MB\n")
    else:
        _log("")

    # ── Mixed-precision scaler (CUDA only) ────────────────────────────
    scaler = build_grad_scaler(device)

    # ── Checkpoint resume ─────────────────────────────────────────────
    ckpt_dir   = os.path.join(cfg["output"]["checkpoint_dir"], run_name)
    resume_path = os.path.join(ckpt_dir, "latest.pt")
    start_epoch = 1

    # Phase-1 optimizer (rebuilt when phase switches in main loop)
    optimizer = build_optimizer(model, criterion, cfg, phase=1)

    if os.path.exists(resume_path):
        start_epoch = load_checkpoint(
            resume_path, model, optimizer, scaler, criterion, device
        )
        # Re-determine correct phase for the resumed epoch
        if start_epoch > frozen_epochs:
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, criterion, cfg, phase=2)

    # ── Log file ──────────────────────────────────────────────────────
    log_dir  = cfg["output"].get("log_dir", "outputs/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,phase,train_loss,val_loss,val_eer,val_gap,epoch_time_s\n")

    # Open-set split: ArcFace val_loss is not a ranking metric. Pick best.pt
    # by verification EER on val identities (same protocol as test eval).
    best_val_eer = float("inf")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()

        # ── Phase transition: unfreeze backbone after frozen_epochs ───
        if epoch == frozen_epochs + 1:
            _log(f"\n  [phase 2] Unfreezing backbone — switching to fine-tune LR\n")
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, criterion, cfg, phase=2)

        phase = 1 if epoch <= frozen_epochs else 2

        # ── LR scheduling: cosine annealing within each phase ─────────
        # (simple linear decay as a lightweight alternative — replace with
        # a proper scheduler if you want cosine warmup restarts)
        if cfg["training"].get("lr_decay", False):
            phase_start = 1 if phase == 1 else frozen_epochs + 1
            phase_len   = frozen_epochs if phase == 1 else total_epochs - frozen_epochs
            phase_epoch = epoch - phase_start
            decay = 1.0 - 0.9 * (phase_epoch / max(phase_len - 1, 1))
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * decay

        # ── Train + Validate ──────────────────────────────────────────
        _log(f"\n  >>> Starting epoch {epoch}/{total_epochs}  (phase {phase})")
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device,
            epoch=epoch, total_epochs=total_epochs, log_interval=log_interval,
        )
        _log(f"  >>> Validating epoch {epoch}/{total_epochs}")
        val_loss   = validate(
            model, criterion, val_loader, device,
            log_interval=max(log_interval, 100),
        )
        _log("  >>> Val verification (class_id pairs, not ArcFace labels)")
        ver = verification_metrics(
            model, val_loader.dataset, device, batch_size=batch_size * 2
        )
        elapsed    = time.time() - t0

        _log(f"  Epoch {epoch:03d}/{total_epochs}  phase={phase}  "
             f"train={train_loss:.4f}  val_arcface={val_loss:.4f}  "
             f"val_eer={ver['eer']:.4f}  gap={ver['gap']:.4f}  "
             f"time={format_time(elapsed)}")

        # ── Logging ───────────────────────────────────────────────────
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{phase},{train_loss:.6f},{val_loss:.6f},"
                f"{ver['eer']:.6f},{ver['gap']:.6f},{elapsed:.1f}\n"
            )

        # ── Checkpointing ─────────────────────────────────────────────
        state = {
            "epoch":      epoch,
            "backbone":   backbone_name,
            "loss_type":  loss_type,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scaler":     scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_eer":    ver["eer"],
            "val_gap":    ver["gap"],
        }
        if loss_type == "arcface":
            state["criterion"] = criterion.loss_fn.state_dict()

        # Always save latest (for resumption)
        save_checkpoint(state, resume_path)

        if ver["eer"] < best_val_eer:
            best_val_eer = ver["eer"]
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(state, best_path)
            _log(f"  [best] new best val_eer={best_val_eer:.4f}  (epoch {epoch})")

    _log(f"\n  Training complete. Best val_eer={best_val_eer:.4f}")
    _log(f"  Logs  : {log_path}")
    _log(f"  Ckpts : {ckpt_dir}/\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train fingerprint embedding model")
    parser.add_argument("--config",   default="src/training/config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--backbone", default=None,
                        choices=["resnet18", "densenet121", "vgg16"],
                        help="Override backbone from config")
    parser.add_argument("--loss",     default=None,
                        choices=["arcface", "triplet"],
                        help="Override loss from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, backbone_override=args.backbone, loss_override=args.loss)


if __name__ == "__main__":
    main()
