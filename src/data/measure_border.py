"""
Measures the fixed scanner-bezel border thickness on each edge and writes it
into config/preprocessing_config.yaml. Meant to be called once from
build_metadata.py (or standalone) - not re-run per image, since the border
is consistent across the dataset.
"""
import cv2
import numpy as np

from src.data.config import load_config, save_config, DEFAULT_CONFIG_PATH


def detect_border(img, flat_std=5):
    """
    Scan inward from each edge; count pixels while the row/column is
    near-uniform (low std) - i.e. flat border, whether it's a dark bezel or
    a light margin. Stops at the first row/column with real variance, which
    is what actual ridge content looks like (always alternating dark ridges
    and light valleys, never flat) regardless of the border's own color.
    """
    h, w = img.shape

    def scan(get_line, length):
        count = 0
        for i in range(length):
            line = get_line(i)
            if np.std(line) < flat_std:
                count += 1
            else:
                break
        return count

    top = scan(lambda i: img[i, :], h)
    bottom = scan(lambda i: img[h - 1 - i, :], h)
    left = scan(lambda i: img[:, i], w)
    right = scan(lambda i: img[:, w - 1 - i], w)
    return top, bottom, left, right


def measure_and_update_config(filepaths, config_path=DEFAULT_CONFIG_PATH,
                               n_samples=None, percentile=98, safety_margin=3,
                               seed=1, verbose=True):
    """
    Measure border thickness over the real images and write a robust
    estimate (a high percentile, not the hard max) plus a small safety
    margin into the config file.

    percentile: using the max here is fragile - a single anomalous image
        (scanner glitch, unusual capture) can pull the max far above what's
        typical, over-trimming real print content for the rest of the
        dataset. A high percentile (98th by default) is generous enough to
        cover the vast majority of images while not being dictated by one
        or two outliers.
    n_samples: None (default) measures over ALL given filepaths - cheap since
        border detection is fast. Pass a number to subsample instead.
    safety_margin: extra pixels trimmed beyond the measured percentile.
    """
    rng = np.random.default_rng(seed)
    filepaths = list(filepaths)
    if n_samples is not None and n_samples < len(filepaths):
        filepaths = list(rng.choice(filepaths, size=n_samples, replace=False))

    results = []
    for path in filepaths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        results.append(detect_border(img))
    results = np.array(results)

    robust_values = np.percentile(results, percentile, axis=0)
    top, bottom, left, right = (robust_values + safety_margin).astype(int).tolist()

    config = load_config(config_path)
    config["border"] = {"top": int(top), "bottom": int(bottom),
                         "left": int(left), "right": int(right)}
    save_config(config, config_path)

    if verbose:
        print(f"Measured border over {len(filepaths)} images "
              f"(p{percentile} + {safety_margin}px margin applied; "
              f"max shown separately to flag outliers):")
        for i, label in enumerate(["top", "bottom", "left", "right"]):
            col = results[:, i]
            print(f"  {label:6s}: p50={np.percentile(col,50):.0f}  "
                  f"p{percentile}={np.percentile(col,percentile):.0f}  max={col.max()}"
                  f"{'  <- outlier gap, good thing we are not using max' if col.max() > np.percentile(col,percentile) + 5 else ''}")
        print(f"Saved border={config['border']} to {config_path}")

    return config["border"]


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/splits/train.csv")
    real_paths = df.loc[df["alteration"] == "real", "filepath"]
    measure_and_update_config(real_paths)