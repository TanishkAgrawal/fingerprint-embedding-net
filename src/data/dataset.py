"""
PyTorch Dataset and DataLoader factory for the fingerprint embedding project.

Loads preprocessed .npy images (128x128 float32, cached by preprocessing.py)
from the split CSVs produced by build_metadata.py -> preprocessing.py.

All three backbones (ResNet18, DenseNet121, VGG16) share the same DataLoader:
  - Input shape  : (3, 128, 128)  - grayscale replicated across 3 channels
  - Value range  : ImageNet-normalized (mean/std applied after [0,1] scaling)
  - class_label  : remapped to a contiguous 0..N-1 range *within* each split,
                   because ArcFace's num_classes must equal the number of
                   unique identities actually present in the training set.

Typical usage
-------------
    from src.data.dataset import get_dataloader

    train_loader, num_classes = get_dataloader(
        'data/processed/train/train.csv', split='train', batch_size=32,
        augment_cfg=cfg['augmentation'],
    )
    val_loader, _ = get_dataloader(
        'data/processed/val/val.csv',   split='val',   batch_size=64
    )
"""

import inspect

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from src.data.preprocessing import augment as _augment_fn

# ImageNet normalization constants - all three pretrained backbones were
# trained with these; applying them here keeps the weight initialisation valid.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class FingerprintDataset(Dataset):
    """
    Dataset that serves cached preprocessed fingerprint images.

    Parameters
    ----------
    csv_path : str
        Path to a processed split CSV (must contain 'processed_path' and
        'class_label' columns - written by preprocessing.cache_processed_dataset).
    augment_images : bool
        If True, apply random augmentation to each image at load time.
        Should be True only for the training split; val/test get
        deterministic preprocessing.
    alteration_filter : str | list[str] | None
        Keep only rows whose 'alteration' column matches this value (or list
        of values).  Useful for ablations:
          - None        -> all rows (real + all altered types)
          - 'real'      -> real images only
          - ['real', 'central_rotation'] -> real + one alteration type
    augment_cfg : dict | None
        Augmentation settings (the 'augmentation:' section of config.yaml),
        forwarded to preprocessing.augment(img, rng, cfg=...). Ignored when
        augment_images is False.
    seed : int | None
        Seed for the augmentation RNG. If None, it is derived from the torch
        seed (torch.manual_seed), so a seeded training run is reproducible.
    """

    def __init__(self, csv_path: str, augment_images: bool = False,
                 alteration_filter=None, augment_cfg=None, seed=None):
        self.df = pd.read_csv(csv_path)
        self.augment_images = augment_images

        if alteration_filter is not None:
            if isinstance(alteration_filter, str):
                alteration_filter = [alteration_filter]
            self.df = (
                self.df[self.df["alteration"].isin(alteration_filter)]
                .reset_index(drop=True)
            )

        # Remap class_label to a contiguous 0..N-1 range for this split.
        # The global codes from build_metadata are not contiguous per-split,
        # which would leave gaps that confuse ArcFace / CrossEntropyLoss.
        unique_labels = sorted(self.df["class_label"].unique())
        self._label_map = {orig: new for new, orig in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

        # Plain Python lists are much cheaper to index per item than a pandas
        # row lookup (df.iloc) - this runs ~40k times per epoch.
        self._paths  = self.df["processed_path"].tolist()
        self._labels = [self._label_map[int(c)] for c in self.df["class_label"]]

        # Augmentation settings. If preprocessing.augment() has not been
        # updated to accept `cfg`, say so instead of silently ignoring them.
        self._augment_kwargs = {}
        if augment_images and augment_cfg:
            if "cfg" in inspect.signature(_augment_fn).parameters:
                self._augment_kwargs["cfg"] = augment_cfg
            else:
                print("  WARNING: preprocessing.augment() has no 'cfg' parameter, so the "
                      "augmentation settings are IGNORED (old augmentation is used).",
                      flush=True)

        # Each worker gets its own RNG (see worker_init_fn below). In the main
        # process the RNG follows the torch seed, so seeded runs repeat.
        if seed is None:
            seed = torch.initial_seed() % (2 ** 32)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        # Load cached 128x128 float32 array in [0, 1]
        img: np.ndarray = np.load(self._paths[idx])   # shape (H, W)

        if self.augment_images:
            img = _augment_fn(img, rng=self._rng, **self._augment_kwargs)

        # (H, W) -> (1, H, W) -> (3, H, W)  by channel replication.
        # All three backbones expect 3-channel input and use ImageNet weights;
        # replication is the standard approach for grayscale -> pretrained RGB
        # networks (equivalent to "Option A" in the project guide).
        img_t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)

        # ImageNet normalization
        img_t = TF.normalize(img_t, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

        return img_t, self._labels[idx]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def original_class_labels(self) -> list:
        """Global class_label codes (from build_metadata) in this split."""
        return sorted(self._label_map.keys())

    def get_class_id_for_label(self, local_label: int) -> str:
        """Map a local (remapped) label back to a human-readable class_id."""
        orig = self.original_class_labels[local_label]
        row = self.df[self.df["class_label"] == orig].iloc[0]
        return row["class_id"]


# ------------------------------------------------------------------------------
# Worker RNG seeding
# ------------------------------------------------------------------------------

def worker_init_fn(worker_id: int) -> None:
    """
    Re-seed each DataLoader worker's numpy RNG independently so that
    augmentation across workers is not correlated.
    Without this, every worker spawned from the same fork inherits the same
    RNG state and produces identical random augmentations.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # worker_info.seed is a unique value per worker (derived from the torch
    # seed); mask to 32-bit to keep it reasonable for numpy's default_rng.
    dataset._rng = np.random.default_rng(worker_info.seed % (2 ** 32))


# ------------------------------------------------------------------------------
# Public factory function
# ------------------------------------------------------------------------------

def get_dataloader(
    csv_path: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    alteration_filter=None,
    pin_memory: bool = True,
    augment_cfg=None,
) -> tuple[DataLoader, int]:
    """
    Build a DataLoader for a given split CSV.

    Parameters
    ----------
    csv_path : str
        Path to a processed split CSV (output of preprocessing.py).
    split : {'train', 'val', 'test'}
        Controls augmentation (train only) and shuffling (train only).
    batch_size : int
        Images per batch.
    num_workers : int
        Parallel data-loading workers.
        On Windows, set to 0 if you hit multiprocessing issues when calling
        from a script that lacks the ``if __name__ == '__main__':`` guard.
    alteration_filter : str | list[str] | None
        Passed through to FingerprintDataset to subset by alteration type.
    pin_memory : bool
        Pin CPU buffers for faster GPU transfer.  Automatically disabled if
        CUDA is not available.
    augment_cfg : dict | None
        Augmentation settings for the training split (see FingerprintDataset).
        Has no effect for val/test.

    Returns
    -------
    loader : DataLoader
    num_classes : int
        Number of unique finger identities in this split - pass this to the
        ArcFace loss as ``num_classes``.
    """
    is_train = (split == "train")

    dataset = FingerprintDataset(
        csv_path=csv_path,
        augment_images=is_train,
        alteration_filter=alteration_filter,
        augment_cfg=augment_cfg if is_train else None,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        # persistent_workers keeps worker processes alive across epochs,
        # saving the fork overhead - meaningful with large num_workers.
        persistent_workers=(num_workers > 0),
        # Drop the last (possibly tiny) batch during training:
        # ArcFace with a near-empty batch can produce degenerate gradients.
        drop_last=is_train,
    )

    return loader, dataset.num_classes