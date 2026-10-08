"""
Step 1-4 of the preprocessing pipeline: load file paths, extract labels from
filenames, define the identity/class label, and split by subject_id into
train/val/test with no leakage.

Expects the SOCOFing folder layout:
    <root>/Real/*.BMP
    <root>/Altered/Altered-Easy/*.BMP
    <root>/Altered/Altered-Medium/*.BMP
    <root>/Altered/Altered-Hard/*.BMP

Usage:
    python build_metadata.py --root data/raw/SOCOFing --out data/splits
"""
import argparse
import os
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.measure_border import measure_and_update_config
from src.data.config import DEFAULT_CONFIG_PATH

# SOCOFing filename formats:
#   Real:    "100__M_Left_thumb_finger.BMP"
#   Altered: "100__M_Left_thumb_finger_CR.BMP"   (CR / Obl / Zcut suffix)
FILENAME_RE = re.compile(
    r"(?P<subject_id>\d+)__(?P<gender>[MF])_(?P<hand>Left|Right)_"
    r"(?P<finger>\w+)_finger(?:_(?P<alteration>Obl|CR|Zcut))?\.BMP",
    re.IGNORECASE,
)

ALTERATION_NAMES = {"Obl": "obliteration", "CR": "central_rotation", "Zcut": "zcut"}


def parse_filename(fname):
    """Extract subject_id, gender, hand, finger, alteration from one filename."""
    match = FILENAME_RE.match(fname)
    if not match:
        return None
    d = match.groupdict()
    return {
        "subject_id": int(d["subject_id"]),
        "gender": d["gender"],
        "hand": d["hand"],
        "finger": d["finger"],
        "alteration": ALTERATION_NAMES.get(d["alteration"], "real") if d["alteration"] else "real",
    }


def build_metadata(root_dir):
    """Step 1 (load) + Step 2 (extract labels) -> one metadata DataFrame."""
    records = []

    # Real images
    real_dir = os.path.join(root_dir, "Real")
    for fname in os.listdir(real_dir):
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        parsed["filepath"] = os.path.join(real_dir, fname)
        parsed["difficulty"] = "none"
        records.append(parsed)

    # Altered images (three difficulty folders)
    for difficulty in ["Easy", "Medium", "Hard"]:
        alt_dir = os.path.join(root_dir, "Altered", f"Altered-{difficulty}")
        for fname in os.listdir(alt_dir):
            parsed = parse_filename(fname)
            if parsed is None:
                continue
            parsed["filepath"] = os.path.join(alt_dir, fname)
            parsed["difficulty"] = difficulty.lower()
            records.append(parsed)

    df = pd.DataFrame(records)

    # Step 3: define the identity/class label.
    # A "class" for metric learning is one specific finger, not one person -
    # ten fingers per subject = ten different classes.
    df["class_id"] = (
        df["subject_id"].astype(str) + "_" + df["hand"] + "_" + df["finger"]
    )
    df["class_label"] = df["class_id"].astype("category").cat.codes

    return df


def split_by_subject(df, val_size=0.15, test_size=0.15, seed=42):
    """
    Step 4: group by subject_id and split so an entire subject (all fingers,
    real + altered) lands in exactly one split - no leakage across splits.
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["subject_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    relative_test_size = test_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["subject_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to the SOCOFing folder")
    parser.add_argument("--out", required=True, help="Output folder for split CSVs")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                         help="Path to write the measured border into")
    parser.add_argument("--skip-border-measure", action="store_true",
                         help="Skip border measurement (e.g. if config already has it set)")
    args = parser.parse_args()

    df = build_metadata(args.root)
    print(f"Total images: {len(df)}  |  Unique subjects: {df['subject_id'].nunique()}  "
          f"|  Unique classes (finger identities): {df['class_id'].nunique()}")

    if not args.skip_border_measure:
        real_paths = df.loc[df["alteration"] == "real", "filepath"]
        measure_and_update_config(real_paths, config_path=args.config)

    train_df, val_df, test_df = split_by_subject(df, args.val_size, args.test_size, args.seed)

    # Sanity check: confirm no subject leaks across splits
    train_subjects = set(train_df["subject_id"])
    val_subjects = set(val_df["subject_id"])
    test_subjects = set(test_df["subject_id"])
    assert not (train_subjects & val_subjects), "Leakage between train and val!"
    assert not (train_subjects & test_subjects), "Leakage between train and test!"
    assert not (val_subjects & test_subjects), "Leakage between val and test!"

    os.makedirs(args.out, exist_ok=True)
    train_df.to_csv(os.path.join(args.out, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out, "test.csv"), index=False)

    print(f"train: {len(train_df)} images, {train_df['subject_id'].nunique()} subjects")
    print(f"val:   {len(val_df)} images, {val_df['subject_id'].nunique()} subjects")
    print(f"test:  {len(test_df)} images, {test_df['subject_id'].nunique()} subjects")


if __name__ == "__main__":
    main()