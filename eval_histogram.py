
"""Plot genuine vs impostor cosine scores saved by src.training.eval."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Read the NPZ file path from the command line
    parser = argparse.ArgumentParser(
        description="Plot genuine and impostor cosine similarity scores."
    )

    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("outputs/logs/eval_densenet121_arcface_full_e25.npz"),
        help="Path to the evaluation NPZ file"
    )

    args = parser.parse_args()
    NPZ = args.scores

    # Check whether the evaluation results exist
    if not NPZ.exists():
        raise SystemExit(
            f"File not found: {NPZ}\n"
            "Run eval first:\n"
            "python -m src.training.eval --config src/training/config.yaml"
        )

    # Load genuine and impostor similarity scores
    data = np.load(NPZ)

    genuine = data["genuine"]
    impostor = data["impostor"]

    # Print evaluation statistics
    print("file     :", NPZ)
    print("genuine  :", genuine.shape, "mean", f"{genuine.mean():.4f}")
    print("impostor :", impostor.shape, "mean", f"{impostor.mean():.4f}")
    print("gap      :", f"{genuine.mean() - impostor.mean():.4f}")

    # Plot similarity distributions
    plt.figure(figsize=(8, 4.5))

    plt.hist(
        impostor,
        bins=40,
        alpha=0.6,
        label="impostor (different finger)"
    )

    plt.hist(
        genuine,
        bins=40,
        alpha=0.6,
        label="genuine (same finger)"
    )

    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.title("Fingerprint embedding verification scores")
    plt.legend()
    plt.tight_layout()

    # Save the histogram as a PNG
    output_path = NPZ.with_suffix(".png")
    plt.savefig(output_path, dpi=300)

    print("Histogram saved to:", output_path)

    # Display the histogram
    plt.show()


if __name__ == "__main__":
    main()
