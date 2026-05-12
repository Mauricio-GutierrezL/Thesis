from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main() -> None:
    labels = [
        "MNIST PCA",
        "MNIST PCA Small",
        "Two Curves",
        "Hidden Manifold",
        "Hidden Manifold Diff",
        "Hyperplanes Diff",
        "Two Curves Diff",
        "Linearly Separable",
    ]
    ccc = np.array([0.850, 0.815, 0.797, 0.752, 0.723, 0.687, 0.681, 0.573])
    halfsep = np.array([0.837, 0.807, 0.794, 0.735, 0.699, 0.668, 0.662, 0.582])
    gap_diff = np.array([0.003, 0.003, 0.002, 0.000, 0.001, 0.005, 0.012, 0.066])

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax1 = plt.subplots(figsize=(10.5, 5.8))

    x = np.arange(len(labels))
    width = 0.34

    ax1.bar(x - width / 2, ccc, width=width, label="CCC", color="#377eb8")
    ax1.bar(x + width / 2, halfsep, width=width, label="Half-Sep", color="#4daf4a")
    ax1.set_ylabel("Mean test accuracy")
    ax1.set_ylim(0.5, 0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        gap_diff,
        color="#e66101",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="CCC/Half-Sep gap diff",
    )
    ax2.set_ylabel("CCC/Half-Sep gap diff")
    ax2.set_ylim(0.0, 0.07)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True)

    sns.despine(ax=ax1, right=False)
    plt.tight_layout()
    outpath = FIGURES_DIR / "ccc-halfsep-gap-bars.png"
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
