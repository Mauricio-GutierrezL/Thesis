from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def build_dataframe() -> pd.DataFrame:
    rows = [
        {
            "family": "Linearly Separable",
            "ccc_test": 0.573,
            "ccc_gap": 0.130,
            "half_gap": 0.064,
        },
        {
            "family": "Hidden Manifold",
            "ccc_test": 0.752,
            "ccc_gap": 0.075,
            "half_gap": 0.075,
        },
        {
            "family": "Hidden Manifold Diff",
            "ccc_test": 0.723,
            "ccc_gap": 0.084,
            "half_gap": 0.083,
        },
        {
            "family": "Hyperplanes Diff",
            "ccc_test": 0.687,
            "ccc_gap": 0.012,
            "half_gap": 0.017,
        },
        {
            "family": "MNIST PCA",
            "ccc_test": 0.850,
            "ccc_gap": -0.000,
            "half_gap": -0.003,
        },
        {
            "family": "MNIST PCA Small",
            "ccc_test": 0.815,
            "ccc_gap": 0.053,
            "half_gap": 0.050,
        },
        {
            "family": "Two Curves",
            "ccc_test": 0.797,
            "ccc_gap": 0.026,
            "half_gap": 0.024,
        },
        {
            "family": "Two Curves Diff",
            "ccc_test": 0.681,
            "ccc_gap": 0.051,
            "half_gap": 0.063,
        },
    ]

    df = pd.DataFrame(rows)
    df["gap_diff_abs"] = (df["ccc_gap"] - df["half_gap"]).abs()
    return df


def add_labels(ax: plt.Axes, df: pd.DataFrame) -> None:
    offsets = {
        "Linearly Separable": (6, 2),
        "Hidden Manifold": (6, 6),
        "Hidden Manifold Diff": (6, -10),
        "Hyperplanes Diff": (6, 6),
        "MNIST PCA": (6, 6),
        "MNIST PCA Small": (6, 6),
        "Two Curves": (6, -10),
        "Two Curves Diff": (6, 6),
    }

    for _, row in df.iterrows():
        dx, dy = offsets.get(row["family"], (6, 6))
        ax.annotate(
            row["family"],
            (row["ccc_test"], row["gap_diff_abs"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left",
            fontsize=10,
        )


def main() -> None:
    df = build_dataframe()
    corr = float(df["ccc_test"].corr(df["gap_diff_abs"]))

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    sns.regplot(
        data=df,
        x="ccc_test",
        y="gap_diff_abs",
        ax=ax,
        scatter_kws={"s": 95, "color": "#1f77b4", "edgecolor": "white", "linewidths": 0.8},
        line_kws={"color": "#d95f02", "linewidth": 2.0},
        ci=None,
    )

    add_labels(ax, df)

    ax.set_xlabel("CCC mean test accuracy")
    ax.set_ylabel(r"$|$ train-test gap difference $|$" + "\n" + r"$= |\,(gap_{CCC} - gap_{Half\text{-}Sep})\,|$")
    ax.set_title(f"Benchmark-family trend between CCC performance and CCC/Half-Sep gap difference (r = {corr:.2f})")
    ax.set_xlim(0.55, 0.88)
    ax.set_ylim(-0.002, 0.072)

    plt.tight_layout()
    outpath = FIGURES_DIR / "gap-vs-ccc-test-acc.png"
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
