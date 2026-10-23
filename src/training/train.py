"""
Shared training loop for the fingerprint embedding project.

Works identically for all three backbones (resnet18, densenet121, vgg16) and
both loss functions (arcface, triplet). All hyperparameters come from a YAML
config so nothing is hardcoded here.

Training strategy (progressive unfreezing):
  Phase 1 - Frozen backbone (config.training.frozen_epochs epochs)
      Only the embedding head is trained. Fast, low memory.
  Phase 2 - Full fine-tuning (remaining epochs)
      Backbone unfreezes with backbone_lr_scale x base_lr. Loss module
      (ArcFace weight matrix) also included in the optimizer.

Learning-rate schedule: linear decay to 10% inside each phase (applied once
per epoch), restarting at the start of phase 2. There is no warm-up.
"""

import argparse
import contextlib
import copy
import inspect
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.dataset import get_dataloader
from src.models.backbone import FingerprintEmbeddingNet
from src.models.losses import get_loss
from src.training.eval import verification_metrics


# -- Helpers ------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print immediately so long epochs don't look hung in the terminal."""
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    _log(f"  [ckpt] saved -> {path}")


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


# -- Epoch functions ----------------------------------------------------------

def train_one_epoch(
    model, criterion, loader, optimizer, scaler, device,
    epoch: int, total_epochs: int, log_interval: int = 50,
    grad_clip_norm: float = 5.0,
):
    model.train()
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()
    total_loss, n_batches = 0.0, 0
    n_total = len(loader)
    pin = device.type == "cuda"

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=pin)
        labels = labels.to(device, non_blocking=pin)

        optimizer.zero_grad(set_to_none=True)

        with _autocast(device):
            embeddings = model(imgs)
        # Loss in float32: ArcFace works with angles/margins near +-1, which is
        # less stable in float16 even under autocast.
        loss = criterion(embeddings.float(), labels)

        scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            # Clipping prevents exploding gradients, especially in the first
            # batches of phase 2 when the backbone unfreezes.
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
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
    """
    ArcFace loss on the val split. NOTE: val fingers are unseen and their
    labels are remapped per split, so this is NOT a meaningful generalisation
    metric. Kept only for completeness (training.compute_val_loss: true).
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    n_total = len(loader)
    pin = device.type == "cuda"

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=pin)
        labels = labels.to(device, non_blocking=pin)

        with _autocast(device):
            embeddings = model(imgs)
        loss = criterion(embeddings.float(), labels)

        total_loss += loss.item()
        n_batches  += 1

        step = batch_idx + 1
        if step == 1 or step % log_interval == 0 or step == n_total:
            avg = total_loss / n_batches
            _log(f"    val {step}/{n_total}  avg_loss={avg:.4f}")

    return total_loss / max(n_batches, 1)


# -- Optimizer builder --------------------------------------------------------

def build_optimizer(model, criterion, cfg: dict, phase: int):
    """
    Build an AdamW optimizer for the given training phase.

    Phase 1 (frozen backbone): only the embedding head + ArcFace matrix.
    Phase 2 (full fine-tune) : backbone (lower LR) + head + ArcFace matrix.
    training.phase2_lr_scale multiplies every learning rate in phase 2.
    """
    base_lr      = cfg["training"]["learning_rate"]
    bb_scale     = cfg["training"]["backbone_lr_scale"]
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    p2_scale     = cfg["training"].get("phase2_lr_scale", 1.0)
    loss_type    = cfg["loss"]["type"]

    if phase == 1:
        # Only the embedding head (backbone is frozen -> its params have
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

    if phase == 2 and p2_scale != 1.0:
        for pg in param_groups:
            pg["lr"] = pg["lr"] * p2_scale

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]
    return optimizer


# -- Main training function ---------------------------------------------------

def train(cfg: dict, backbone_override: str | None = None,
          loss_override: str | None = None, run_name_override: str | None = None,
          no_resume: bool = False, resume_baseline: bool = False):

    # -- Config resolution ----------------------------------------------
    tcfg          = cfg["training"]
    backbone_name = backbone_override or cfg["model"]["backbone"]
    loss_type     = loss_override     or cfg["loss"]["type"]
    embedding_dim = cfg["model"]["embedding_dim"]
    total_epochs  = tcfg["epochs"]
    frozen_epochs = tcfg["frozen_epochs"]
    batch_size    = tcfg["batch_size"]
    num_workers   = cfg["data"].get("num_workers", 0)
    log_interval  = tcfg.get("log_interval", 50)
    grad_clip     = float(tcfg.get("grad_clip_norm", 5.0))
    patience      = int(tcfg.get("early_stop_patience", 0))
    val_loss_on   = bool(tcfg.get("compute_val_loss", False))
    seed          = int(tcfg.get("seed", 42))

    set_seed(seed)
    run_t0 = time.time()

    # Run identifier used for checkpoints + logs
    run_name = run_name_override or f"{backbone_name}_{loss_type}"
    protected = {
        "resnet18_arcface",
        "densenet_arcface",
        "densenet121_arcface",
        "vgg16_arcface",
    }
    if run_name in protected and not resume_baseline:
        raise SystemExit(
            f"Refusing to train into protected folder '{run_name}'.\n"
            "That checkpoint is a finished baseline (ResNet/DenseNet/VGG).\n"
            f"New experiment:  --run-name {run_name}_v2 --no-resume\n"
            "Continue VGG-14: --backbone vgg16 --run-name vgg16_arcface --resume-baseline"
        )

    # -- Device ----------------------------------------------------------
    device = resolve_device(cfg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # fixed input size -> faster convs
    _log(f"\n{'='*60}")
    _log(f"  Run      : {run_name}")
    _log(f"  Python   : {sys.executable}")
    _log(f"  PyTorch  : {torch.__version__}")
    _log(f"  Device   : {device}")
    if device.type == "cuda":
        idx = device.index or 0
        _log(f"  GPU      : {torch.cuda.get_device_name(idx)}")
        _log("  Task Mgr : NVIDIA graph at 0% is normal if it shows the 3D engine.")
        _log("             Click GPU 1 (NVIDIA) -> change the graph to CUDA / Compute.")
        _log("             AMD Radeon is the iGPU and is not used by PyTorch.")
    elif tcfg.get("device", "cuda").lower() == "auto":
        _log("  NOTE     : CUDA not available - running on CPU (training.device: auto).")
    _log(f"  Epochs   : {total_epochs}  (frozen={frozen_epochs}, "
         f"finetune={total_epochs - frozen_epochs})  early_stop_patience={patience or 'off'}")
    _log(f"  Log every: {log_interval} train batches")
    _log(f"  HP       : lr={tcfg['learning_rate']}  "
         f"bb_scale={tcfg['backbone_lr_scale']}  wd={tcfg.get('weight_decay')}  "
         f"batch={batch_size}  dim={embedding_dim}  lr_decay={tcfg.get('lr_decay')}")
    _log(f"  HP       : clip={grad_clip}  phase2_lr_scale={tcfg.get('phase2_lr_scale', 1.0)}  "
         f"seed={seed}  val_loss_pass={val_loss_on}")
    _log(f"  Loss     : {loss_type}  {cfg['loss'].get('kwargs', {})}")
    _log(f"{'='*60}\n")

    # -- Data ------------------------------------------------------------
    train_csv  = cfg["data"]["train_csv"]
    val_csv    = cfg["data"]["val_csv"]
    alt_filter = cfg["data"].get("alteration_filter", None)

    # Augmentation settings are passed to the loader only if get_dataloader
    # supports them. If not, say so loudly: otherwise trials that differ only
    # in augmentation would silently be identical.
    loader_kwargs = {}
    aug_cfg = cfg.get("augmentation")
    if aug_cfg is not None:
        if "augment_cfg" in inspect.signature(get_dataloader).parameters:
            loader_kwargs["augment_cfg"] = aug_cfg
        else:
            _log("  WARNING: get_dataloader() has no 'augment_cfg' parameter, so the\n"
                 "           'augmentation:' section of the config is IGNORED.\n")

    train_loader, num_classes = get_dataloader(
        train_csv, split="train", batch_size=batch_size,
        num_workers=num_workers, alteration_filter=alt_filter, **loader_kwargs,
    )
    val_loader, _ = get_dataloader(
        val_csv, split="val", batch_size=batch_size * 2,
        num_workers=num_workers,
    )
    _log(f"  Train batches : {len(train_loader)}  |  classes : {num_classes}")
    _log(f"  Val   batches : {len(val_loader)}\n")

    # -- Model + Loss ----------------------------------------------------
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

    # -- Mixed-precision scaler (CUDA only) -------------------------------
    scaler = build_grad_scaler(device)

    # -- Checkpoint resume -----------------------------------------------
    ckpt_dir    = os.path.join(cfg["output"]["checkpoint_dir"], run_name)
    resume_path = os.path.join(ckpt_dir, "latest.pt")
    _log(f"  Ckpt dir : {ckpt_dir}")
    _log(f"  Resume   : {resume_path}  exists={os.path.exists(resume_path)}")

    ckpt = None
    if os.path.exists(resume_path):
        if no_resume:
            raise SystemExit(
                f"Refusing to train: {resume_path} already exists.\n"
                "Pass a new --run-name so this run cannot overwrite that checkpoint."
            )
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt.get("backbone") and ckpt["backbone"] != backbone_name:
            raise SystemExit(
                f"Checkpoint backbone is '{ckpt['backbone']}', but this run is "
                f"'{backbone_name}'. Use a different --run-name."
            )
        if ckpt["epoch"] + 1 > total_epochs:
            raise SystemExit(
                f"Nothing to do: {resume_path} is already at epoch "
                f"{ckpt['epoch']}, but config training.epochs is {total_epochs}.\n"
                "Raise epochs, or use a new --run-name (and --no-resume) for a fresh run."
            )

    # Build the optimizer for the phase the CHECKPOINT was saved in, so its
    # parameter groups match the saved state (a phase-2 checkpoint has more
    # groups than a phase-1 optimizer). The loop below handles any phase change.
    start_epoch = 1
    bad_epochs  = 0
    saved_phase = 1
    if ckpt is not None:
        saved_phase = ckpt.get("phase", 1 if ckpt["epoch"] <= frozen_epochs else 2)
        if saved_phase == 2:
            model.unfreeze_backbone()
    optimizer = build_optimizer(model, criterion, cfg, phase=saved_phase)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if "criterion" in ckpt and hasattr(criterion, "loss_fn"):
            criterion.loss_fn.load_state_dict(ckpt["criterion"])
        start_epoch = ckpt["epoch"] + 1
        bad_epochs  = ckpt.get("bad_epochs", 0)
        _log(f"  [ckpt] resumed from epoch {ckpt['epoch']}  "
             f"(phase {saved_phase}, val_eer={ckpt.get('val_eer', float('nan')):.4f})")
        del ckpt

    # -- Run metadata (which exact settings produced this checkpoint) ------
    os.makedirs(ckpt_dir, exist_ok=True)
    effective_cfg = copy.deepcopy(cfg)
    effective_cfg["model"]["backbone"] = backbone_name
    effective_cfg["loss"]["type"] = loss_type
    with open(os.path.join(ckpt_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(effective_cfg, f, sort_keys=False)

    # -- Log file ----------------------------------------------------------
    log_dir  = cfg["output"].get("log_dir", "outputs/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,phase,train_loss,val_loss,val_eer,val_gap,epoch_time_s\n")

    # Open-set split: ArcFace val_loss is not a ranking metric. Pick best.pt
    # by verification EER on val identities (same protocol as test eval).
    best_val_eer = float("inf")
    best_epoch   = None
    best_path    = os.path.join(ckpt_dir, "best.pt")
    if start_epoch > 1 and os.path.exists(best_path):
        # Keep the previous best so a resumed epoch cannot clobber it by default.
        prev_best = torch.load(best_path, map_location="cpu", weights_only=False)
        best_val_eer = float(prev_best.get("val_eer", best_val_eer))
        best_epoch   = prev_best.get("epoch")
        del prev_best

    epochs_run    = start_epoch - 1
    stopped_early = False

    # -- Training loop -----------------------------------------------------
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()

        # -- Phase transition: unfreeze backbone after frozen_epochs -------
        if epoch == frozen_epochs + 1:
            _log(f"\n  [phase 2] Unfreezing backbone - switching to fine-tune LR\n")
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, criterion, cfg, phase=2)
            bad_epochs = 0   # early-stopping patience counts from the unfreeze

        phase = 1 if epoch <= frozen_epochs else 2

        # -- LR scheduling: linear decay to 10% within each phase ----------
        # (phase 2 restarts from its own initial LR)
        if tcfg.get("lr_decay", False):
            phase_start = 1 if phase == 1 else frozen_epochs + 1
            phase_len   = frozen_epochs if phase == 1 else total_epochs - frozen_epochs
            phase_epoch = epoch - phase_start
            decay = 1.0 - 0.9 * (phase_epoch / max(phase_len - 1, 1))
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * decay

        # -- Train + Validate ----------------------------------------------
        _log(f"\n  >>> Starting epoch {epoch}/{total_epochs}  (phase {phase})")
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device,
            epoch=epoch, total_epochs=total_epochs, log_interval=log_interval,
            grad_clip_norm=grad_clip,
        )
        if val_loss_on:
            _log(f"  >>> Validating epoch {epoch}/{total_epochs}")
            val_loss = validate(
                model, criterion, val_loader, device,
                log_interval=max(log_interval, 100),
            )
        else:
            val_loss = float("nan")
        _log("  >>> Val verification (class_id pairs, not ArcFace labels)")
        ver = verification_metrics(
            model, val_loader.dataset, device, batch_size=batch_size * 2
        )
        elapsed    = time.time() - t0
        epochs_run = epoch

        _log(f"  Epoch {epoch:03d}/{total_epochs}  phase={phase}  "
             f"train={train_loss:.4f}  val_arcface={val_loss:.4f}  "
             f"val_eer={ver['eer']:.4f}  gap={ver['gap']:.4f}  "
             f"time={format_time(elapsed)}")

        # -- Logging -----------------------------------------------------
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{phase},{train_loss:.6f},{val_loss:.6f},"
                f"{ver['eer']:.6f},{ver['gap']:.6f},{elapsed:.1f}\n"
            )

        # -- Best-model bookkeeping --------------------------------------
        improved = ver["eer"] < best_val_eer
        if improved:
            best_val_eer, best_epoch, bad_epochs = ver["eer"], epoch, 0
        else:
            bad_epochs += 1

        # -- Checkpointing -----------------------------------------------
        state = {
            "epoch":      epoch,
            "phase":      phase,
            "bad_epochs": bad_epochs,
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

        if improved:
            save_checkpoint(state, best_path)
            _log(f"  [best] new best val_eer={best_val_eer:.4f}  (epoch {epoch})")

        # -- Early stopping (only after the backbone has unfrozen) ---------
        if patience and phase == 2 and bad_epochs >= patience:
            _log(f"\n  [early stop] no val_eer gain for {bad_epochs} epochs "
                 f"(best {best_val_eer:.4f} at epoch {best_epoch})")
            stopped_early = True
            break

    summary = {
        "run_name": run_name,
        "backbone": backbone_name,
        "loss": loss_type,
        "best_val_eer": best_val_eer,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "stopped_early": stopped_early,
        "seed": seed,
        "train_time_s": round(time.time() - run_t0, 1),
    }
    with open(os.path.join(ckpt_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _log(f"\n  Training complete. Best val_eer={best_val_eer:.4f} (epoch {best_epoch})")
    _log(f"  Logs  : {log_path}")
    _log(f"  Ckpts : {ckpt_dir}/\n")
    return {
        "run_name": run_name,
        "ckpt_dir": ckpt_dir,
        "best_val_eer": best_val_eer,
        "best_epoch": best_epoch,
        "log_path": log_path,
    }


# -- CLI entry point ----------------------------------------------------------

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
    parser.add_argument("--run-name", default=None,
                        help="Checkpoint/log folder name (default: {backbone}_{loss})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Abort if checkpoints/{run-name}/latest.pt already exists")
    parser.add_argument("--resume-baseline", action="store_true",
                        help="Allow writing into resnet18_arcface / densenet_arcface / vgg16_arcface")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, backbone_override=args.backbone, loss_override=args.loss,
          run_name_override=args.run_name, no_resume=args.no_resume,
          resume_baseline=args.resume_baseline)


if __name__ == "__main__":
    main()