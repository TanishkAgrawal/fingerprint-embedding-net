import numpy as np, pandas as pd, glob

for split in ["train", "val", "test"]:
    files = glob.glob(f"data/processed/{split}/*.npy")
    n_csv = len(pd.read_csv(f"data/processed/{split}/{split}.csv"))
    bad = 0
    for f in files:
        a = np.load(f)
        if a.shape != (128, 128) or a.dtype != np.float32 or np.isnan(a).any() or a.min() < 0 or a.max() > 1:
            bad += 1
    print(split, "files:", len(files), "csv rows:", n_csv, "bad:", bad)