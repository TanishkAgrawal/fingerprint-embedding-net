import cv2
import numpy as np
import pandas as pd
from src.data.measure_border import detect_border

df = pd.read_csv("data/splits/train.csv")
real_paths = df.loc[df["alteration"] == "real", "filepath"]

results = np.array([detect_border(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in real_paths])
for i, label in enumerate(["top", "bottom", "left", "right"]):
    col = results[:, i]
    print(f"{label:6s}: p50={np.percentile(col,50):.0f}  p95={np.percentile(col,95):.0f}  "
          f"p99={np.percentile(col,99):.0f}  max={col.max()}")