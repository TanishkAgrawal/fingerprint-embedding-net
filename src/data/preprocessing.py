"""
Step 5: per-image preprocessing (segment, enhance, resize, normalize) - run
once and cached to data/processed/, not recomputed every epoch.

Step 6: augmentation lives in a separate function (`augment`) and is applied
on-the-fly at training time, on top of the cached preprocessed image, so
every epoch sees a different random variation.

All tunable settings (border trim, image size, CLAHE params) come from
config/preprocessing_config.yaml via src/data/config.py, rather than being
hardcoded here - build_metadata.py writes the measured border into that file,
and this module (plus later training/eval/demo code) just reads it.
"""
import os

import cv2
import numpy as np
import pandas as pd

from src.data.config import load_config

_CONFIG = load_config()
BORDER = _CONFIG["border"]
IMAGE_CFG = _CONFIG["image"]
CLAHE_CFG = _CONFIG["clahe"]


def trim_border(img, border=None):
    """Remove the fixed scanner bezel. A no-op on any edge left at 0."""
    border = border or BORDER
    h, w = img.shape
    b, t, l, r = border["bottom"], border["top"], border["left"], border["right"]
    return img[t:h - b, l:w - r]


def pad_to_square(img):
    """
    Pad a non-square image to make it square, so the final resize scales
    width and height by the same factor instead of stretching them
    differently. Uses edge-replicate padding (extends the nearest real
    pixels outward) rather than a flat fill color - a computed constant
    color (e.g. the image mean) tends to sit noticeably darker than the
    scan's actual near-white background and shows up as a visible border;
    replicating the edge blends in since the edge itself is background.
    """
    h, w = img.shape
    if h == w:
        return img
    diff = abs(h - w)
    pad1, pad2 = diff // 2, diff - diff // 2
    if h > w:
        return cv2.copyMakeBorder(img, 0, 0, pad1, pad2, cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(img, pad1, pad2, 0, 0, cv2.BORDER_REPLICATE)


def preprocess_fingerprint(img_path, size=None, apply_crop=None, margin=None):
    """
    Segment (optional), enhance, resize, normalize a single fingerprint image.
    Any argument left as None falls back to the value in
    config/preprocessing_config.yaml.

    apply_crop: SOCOFing images (96x103) are already tightly framed around the
        print with almost no background, so cropping is OFF by default there -
        an Otsu-based crop on an already-tight image tends to clip into the
        print itself rather than remove background. Turn this on for datasets
        that do have real surrounding background (e.g. FVC, IIT-Delhi).
    margin: extra pixels kept around the detected fingerprint region so a
        slightly-too-tight mask can't cut into real ridge area.
    """
    size = IMAGE_CFG["size"] if size is None else size
    apply_crop = IMAGE_CFG["apply_crop"] if apply_crop is None else apply_crop
    margin = IMAGE_CFG["crop_margin"] if margin is None else margin

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = trim_border(img)

    # Denoise before CLAHE: a median filter removes faint sensor noise while
    # preserving ridge edges. Skipping this step is what caused CLAHE to
    # amplify near-invisible background noise into a visible speckled border.
    k = CLAHE_CFG.get("denoise_kernel", 0)
    if k and k > 1:
        img = cv2.medianBlur(img, k)

    # Ridge enhancement
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CFG["clip_limit"],
                             tileGridSize=tuple(CLAHE_CFG["tile_grid_size"]))
    img = clahe.apply(img)

    if apply_crop:
        # ROI crop: remove background via Otsu threshold on the inverted image
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Faint edge ridges can get misclassified as background by a single
        # global Otsu threshold, leaving small gaps/holes in the mask right at
        # the print's boundary. Morphological closing merges those gaps back
        # in before we measure the bounding box, so the box isn't drawn
        # smaller than the actual print.
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Safety margin: expand the box outward so a still-slightly-tight
            # mask can't clip into real fingerprint area.
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1 = min(img.shape[1], x + w + margin)
            y1 = min(img.shape[0], y + h + margin)
            img = img[y0:y1, x0:x1]

    img = pad_to_square(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return img


def cache_processed_dataset(metadata_csv, output_dir, size=None):
    """
    Run preprocess_fingerprint over every row in a split CSV (from
    build_metadata.py) and save the result as .npy files in output_dir.
    Also writes an updated CSV pointing at the cached files.
    """
    df = pd.read_csv(metadata_csv)
    os.makedirs(output_dir, exist_ok=True)

    cached_paths = []
    for i, row in df.iterrows():
        img = preprocess_fingerprint(row["filepath"], size=size)
        out_path = os.path.join(output_dir, f"{i}.npy")
        np.save(out_path, img)
        cached_paths.append(out_path)
        if i % 2000 == 0:
            print(f"  processed {i}/{len(df)}")

    df["processed_path"] = cached_paths
    cached_csv = os.path.join(output_dir, os.path.basename(metadata_csv))
    df.to_csv(cached_csv, index=False)
    print(f"Cached {len(df)} images to {output_dir}, metadata written to {cached_csv}")
    return df


def augment(img, rng=None, cfg=None):
    """
    img: (H, W) float32 in [0, 1]. cfg: the `augmentation` section of config.
    Missing keys fall back to the previous hardcoded behaviour.
    """
    rng = rng or np.random.default_rng()
    cfg = cfg or {}
    h, w = img.shape

    rot = cfg.get("rotation_deg", 20)
    trans = cfg.get("translate_frac", 0.0)
    s_lo, s_hi = cfg.get("scale_range", [1.0, 1.0])
    crop_p = cfg.get("partial_crop_prob", 0.3)
    crop_min = cfg.get("partial_crop_min_frac", 0.7)
    erase_p = cfg.get("erase_prob", 0.0)
    noise_std = cfg.get("noise_std", 0.0)

    # One affine for rotation + scale + translation (single interpolation pass)
    angle = rng.uniform(-rot, rot)
    scale = rng.uniform(s_lo, s_hi)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += rng.uniform(-trans, trans) * w
    M[1, 2] += rng.uniform(-trans, trans) * h
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Contrast / brightness jitter
    img = np.clip(img * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0.0, 1.0)

    # Partial crop, resized back
    if rng.random() < crop_p:
        f = rng.uniform(crop_min, 0.95)
        ch, cw = int(h * f), int(w * f)
        top = rng.integers(0, h - ch + 1)
        left = rng.integers(0, w - cw + 1)
        img = cv2.resize(img[top:top + ch, left:left + cw], (w, h))

    # Random erasing: blank a patch so the net can't rely on one region
    if rng.random() < erase_p:
        area = rng.uniform(0.02, 0.15) * h * w
        aspect = rng.uniform(0.5, 2.0)
        eh = min(h, max(1, int(np.sqrt(area * aspect))))
        ew = min(w, max(1, int(np.sqrt(area / aspect))))
        y0 = rng.integers(0, h - eh + 1)
        x0 = rng.integers(0, w - ew + 1)
        img = img.copy()
        img[y0:y0 + eh, x0:x0 + ew] = np.median(img)

    # Gaussian sensor noise
    if noise_std > 0:
        img = np.clip(img + rng.normal(0, noise_std, img.shape), 0.0, 1.0)

    return img.astype(np.float32)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", required=True, help="e.g. data/splits/train.csv")
    parser.add_argument("--out", required=True, help="e.g. data/processed/train")
    parser.add_argument("--size", type=int, default=None,
                         help="Overrides config image.size if given")
    args = parser.parse_args()

    cache_processed_dataset(args.split_csv, args.out, size=args.size)