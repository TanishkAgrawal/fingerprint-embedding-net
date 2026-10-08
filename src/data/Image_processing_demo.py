import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import matplotlib.pyplot as plt
from src.data.preprocessing import preprocess_fingerprint, pad_to_square
import cv2

img_path = "data/raw/SOCOFing/Real/1__M_Left_index_finger.BMP"
raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw)
padded = pad_to_square(clahe_img)
final = preprocess_fingerprint(img_path, size=128, apply_crop=False)  # match your actual call

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for a, im, title in zip(ax, [raw, clahe_img, padded, final],
                        ["raw", "after CLAHE", "after pad_to_square", "final (resized)"]):
    a.imshow(im, cmap="gray"); a.set_title(f"{title}\n{im.shape}"); a.axis("off")
plt.show()

import pandas as pd, numpy as np, cv2
import matplotlib.pyplot as plt

df = pd.read_csv("data/splits/train.csv")
sample = df.sample(8, random_state=1)

fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for a, path in zip(ax.flat, sample["filepath"]):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    a.imshow(img, cmap="gray"); a.axis("off")
plt.show()