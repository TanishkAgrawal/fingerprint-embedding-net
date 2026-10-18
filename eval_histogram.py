"""Plot genuine vs impostor cosine scores saved by src.training.eval."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NPZ = Path("outputs/logs/eval_resnet18_arcface_full_e25.npz")
# After a full eval, switch to:
# NPZ = Path("outputs/logs/eval_resnet18_arcface_full.npz")


def main() -> None:
    if not NPZ.exists():
        raise SystemExit(
            f"File not found: {NPZ}\n"
            "Run eval first:\n"
            "python -m src.training.eval --config src/training/config.yaml"
        )

    data = np.load(NPZ)
    genuine = data["genuine"]
    impostor = data["impostor"]

    print("file     :", NPZ)
    print("genuine  :", genuine.shape, "mean", f"{genuine.mean():.4f}")
    print("impostor :", impostor.shape, "mean", f"{impostor.mean():.4f}")
    print("gap      :", f"{genuine.mean() - impostor.mean():.4f}")

    plt.figure(figsize=(8, 4.5))
    plt.hist(impostor, bins=40, alpha=0.6, label="impostor (different finger)")
    plt.hist(genuine, bins=40, alpha=0.6, label="genuine (same finger)")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.title("Fingerprint embedding verification scores")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
